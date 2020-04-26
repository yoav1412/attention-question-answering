from keras.callbacks import Callback
from keras.layers import Embedding, Input, Lambda, Dropout, Conv1D, Softmax, Concatenate, Add
from keras.initializers import Constant
from keras.regularizers import l2
from keras.models import Model
import pickle
from keras import backend as K
from layers.encoder_stack import EncoderStack
from layers.highway_network import HighwayNetwork
from layers.context_to_query_attention import ContextQueryAttention
from layers.probs_to_span import ProbsToSpan
import numpy as np
import tensorflow as tf
from conf import Config as cf

def build_model(cf):
    embedding_matrix = np.load(cf.REDUCED_EMBEDDING_MATRIX_PATH)

    # Input:
    context_inp = Input(shape=(cf.MAX_CONTEXT_LENGTH,))
    query_inp = Input(shape=(cf.MAX_QUESTION_LENGTH,))

    # Embedding Layer:
    embedding_layer = Embedding(embedding_matrix.shape[0],
                                embedding_matrix.shape[1],
                                embeddings_initializer=Constant(embedding_matrix),
                                trainable=False, name="EmbeddingLayer")
    context_emb = embedding_layer(context_inp)
    query_emb = embedding_layer(query_inp)

    context_emb = Dropout(rate=cf.BETWEEN_LAYER_DROPOUT_RATE)(context_emb)
    query_emb = Dropout(rate=cf.BETWEEN_LAYER_DROPOUT_RATE)(query_emb)

    c_mask = Lambda(lambda x: tf.cast(tf.cast(x, tf.bool), tf.float32))(context_inp)  # [bs, c_len]
    q_mask = Lambda(lambda x: tf.cast(tf.cast(x, tf.bool), tf.float32))(query_inp)


    # Highway Network:
    highway = HighwayNetwork(filters=cf.D_MODEL, n_layers=2, dropout=cf.BETWEEN_LAYER_DROPOUT_RATE,
                             initializer='glorot_uniform')
    proj_context_emb = highway(context_emb)
    proj_query_emb = highway(query_emb)


    # Embedding Encoding:  output -> (?, seq_len, d_model)
    emb_enc_stack = EncoderStack(n_blocks=cf.N_EMB_ENC_BLOCKS, input_dim=cf.D_MODEL, num_heads=cf.N_HEADS,
                                 output_dim=cf.D_MODEL, n_convs=4, kernel_size=(7,1),
                                 pwffn_filters=[cf.D_MODEL, cf.D_MODEL], dropout=cf.BETWEEN_LAYER_DROPOUT_RATE,
                                 name="EmbEnc")
    context_encoding = emb_enc_stack(proj_context_emb, c_mask)
    query_encoding = emb_enc_stack(proj_query_emb, q_mask)


    # C2Q and Q2C Attention:  output -> (?, seq_len, 4*d_model)
    c2q_attention_layer = ContextQueryAttention(output_dim=cf.D_MODEL * 4, c_maxlen=cf.MAX_CONTEXT_LENGTH,
                                                q_maxlen=cf.MAX_QUESTION_LENGTH, dropout=0, name="BiAttn")
    attn = c2q_attention_layer([context_encoding, query_encoding, c_mask, q_mask])

    attn = Dropout(rate=cf.BETWEEN_LAYER_DROPOUT_RATE)(attn)

    attn = Conv1D(cf.D_MODEL, 1,
           kernel_initializer='glorot_uniform',
           kernel_regularizer=l2(cf.L2_LAMBDA),
           activation='linear')(attn)

    # Model Encoding Stack: output -> (?, seq_len, 4*d_model)

    model_encoder_stack = EncoderStack(n_blocks=cf.N_MOD_ENC_BLOCKS, input_dim=cf.D_MODEL, num_heads=cf.N_HEADS,
                                       output_dim=cf.D_MODEL, n_convs=2, kernel_size=(5,1),
                                       pwffn_filters=[cf.D_MODEL, cf.D_MODEL],
                                       dropout=cf.BETWEEN_LAYER_DROPOUT_RATE, name='modelEnc')
    M0 = model_encoder_stack(attn, c_mask)
    M1 = model_encoder_stack(M0, c_mask)
    M2 = model_encoder_stack(M1, c_mask)



    # Output Layer:
    start = Concatenate(axis=2)([M0, M1])
    start = Conv1D(1, 1, activation='linear', kernel_regularizer=l2(cf.L2_LAMBDA))(start)
    start = Dropout(rate=cf.BETWEEN_LAYER_DROPOUT_RATE)(start)

    # apply mask, as to not predict position in padded section of sequence:
    mult_mask = Lambda(lambda msk: cf.VERY_NEGATIVE_NUMBER * (1 - msk))(c_mask)
    start = Lambda(lambda inp: K.squeeze(inp, axis=2))(start)
    start = Add()([start, mult_mask])
    start = Lambda(lambda inp: K.expand_dims(inp, axis=2))(start)

    p1 = Softmax(axis=1, name="output_SftMxP1")(start)

    # Right Branch (as in paper):
    end = Concatenate(axis=2)([M0, M2])

    end = Conv1D(1, 1, activation='linear', kernel_regularizer=l2(cf.L2_LAMBDA), name="output_layer_linear")(end)
    end = Dropout(rate=cf.BETWEEN_LAYER_DROPOUT_RATE)(end)

    # apply mask, as to not predict position in padded section of sequence: #
    mult_mask = Lambda(lambda msk: cf.VERY_NEGATIVE_NUMBER * (1 - msk))(c_mask)
    end = Lambda(lambda inp: K.squeeze(inp, axis=2))(end)
    end = Add()([end, mult_mask])
    end = Lambda(lambda inp: K.expand_dims(inp, axis=2))(end)

    p2 = Softmax(axis=1, name="output_SftMxP2")(end)

    start_idx, end_idx = ProbsToSpan(name="ProbsToSpan")([p1, p2])

    p1 = Lambda(lambda inp: K.squeeze(inp, axis=2))(p1)
    p2 = Lambda(lambda inp: K.squeeze(inp, axis=2))(p2)

    model = Model(inputs=[context_inp, query_inp], outputs=[p1, p2, start_idx, end_idx])

    return model


def get_squad_data_for_model(path, shuffle_data=True):
    """
    :param path: path to pre-processed data file.
    :param shuffle_data: bool. if True, will randomly shuffle the list of SquadExample objects after reading it.
    :return: the tokenized data and the start/end positions ready for model training / inference.
    """
    if cf.LIMIT_TRAIN is not None:
        data = pickle.load(open(path, 'rb'))[:cf.LIMIT_TRAIN]
    else:
        data = pickle.load(open(path, 'rb'))

    if shuffle_data:
        np.random.shuffle(data)

    tokenized_contexts = np.array([e.context_tokenized for e in data])
    tokenized_questions = np.array([e.question_tokenized for e in data])

    y_start_positions = np.array([e.gt.tokenized_span[0] for e in data])
    y_end_positions = np.array([e.gt.tokenized_span[1] for e in data])

    return tokenized_contexts, tokenized_questions, y_start_positions, y_end_positions

class LearningRateCallback(Callback):
    """
    A Callback in charge of the learning rate schedule, according to attention-question-answering paper section 4.1.1
    """
    def __init__(self):
        self.step = 0
        super(LearningRateCallback, self).__init__()

    def on_train_begin(self, logs=None):
        lr = cf.INITIAL_LR
        K.set_value(self.model.optimizer.lr, lr)

    def on_batch_end(self, batch, logs=None):
        # Warm-up the first 1000 steps
        self.step += 1
        if self.step < cf.LR_WARMUP_STEPS:
            lr = (cf.CONST_LR / np.log(cf.LR_WARMUP_STEPS)) * np.log(self.step)
        else:
            lr = cf.CONST_LR
        K.set_value(self.model.optimizer.lr, lr)
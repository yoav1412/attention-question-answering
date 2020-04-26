from keras.layers import Layer
import tensorflow as tf
from conf import Config as cf
from keras.initializers import VarianceScaling
import keras.backend as K
from keras.regularizers import l2


class ContextQueryAttention(Layer):
    """
    Context-to-query attention.
    See https://github.com/google-research/google-research/blob/master/qanet/squad_helper.py
    """

    def __init__(self, output_dim, c_maxlen, q_maxlen, dropout, **kwargs):
        self.output_dim = output_dim
        self.c_maxlen = c_maxlen
        self.q_maxlen = q_maxlen
        self.dropout = dropout
        super(ContextQueryAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: [(None, ?, 128), (None, ?, 128)]
        init = VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform')
        self.W0 = self.add_weight(name='W0',
                                  shape=(input_shape[0][-1], 1),
                                  initializer=init,
                                  regularizer = l2(cf.L2_LAMBDA),
                                  trainable=True)
        self.W1 = self.add_weight(name='W1',
                                  shape=(input_shape[1][-1], 1),
                                  initializer=init,
                                  regularizer=l2(cf.L2_LAMBDA),
                                  trainable=True)
        self.W2 = self.add_weight(name='W2',
                                  shape=(1, 1, input_shape[0][-1]),
                                  initializer=init,
                                  regularizer=l2(cf.L2_LAMBDA),
                                  trainable=True)
        self.bias = self.add_weight(name='linear_bias',
                                    shape=([1]),
                                    initializer='zero',
                                    regularizer=l2(cf.L2_LAMBDA),
                                    trainable=True)
        super(ContextQueryAttention, self).build(input_shape)

    def mask_logits(self, inputs, mask, mask_value=cf.VERY_NEGATIVE_NUMBER):
        mask = tf.cast(mask, tf.float32)
        return inputs + mask_value * (1 - mask)

    def call(self, inputs):
        x_cont, x_ques, c_mask, q_mask = inputs
        # get similarity matrix S
        subres0 = K.tile(tf.matmul(x_cont, self.W0), [1, 1, self.q_maxlen])
        subres1 = K.tile(K.permute_dimensions(tf.matmul(x_ques, self.W1), pattern=(0, 2, 1)), [1, self.c_maxlen, 1])
        subres2 = K.batch_dot(x_cont * self.W2, K.permute_dimensions(x_ques, pattern=(0, 2, 1)))
        S = subres0 + subres1 + subres2
        S += self.bias
        q_mask = tf.expand_dims(q_mask, 1)
        S_ = tf.nn.softmax(self.mask_logits(S, q_mask))
        c_mask = tf.expand_dims(c_mask, 2)
        S_T = K.permute_dimensions(tf.nn.softmax(self.mask_logits(S, c_mask), axis=1), (0, 2, 1))
        c2q = tf.matmul(S_, x_ques)
        q2c = tf.matmul(tf.matmul(S_, S_T), x_cont)
        result = K.concatenate([x_cont, c2q, x_cont * c2q, x_cont * q2c], axis=-1)

        return result

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)

import os

class Config:
    # Paths:
    DATA_DIR = r"./data"
    SQUAD_DATA_DIR = os.path.join(DATA_DIR, r"squad")
    SQUAD_TRAIN_DATA_PATH = os.path.join(SQUAD_DATA_DIR, r"train-v1.1.json")
    SQUAD_VAL_DATA_PATH = os.path.join(SQUAD_DATA_DIR, r"dev-v1.1.json")
    PARSED_VAL_DATA_PATH = os.path.join(SQUAD_DATA_DIR, r"parsed_val.pickle")
    PARSED_TRAIN_DATA_PATH = os.path.join(SQUAD_DATA_DIR, r"parsed_train.pickle")
    VAL_PREDICTIONS_DIR = os.path.join(DATA_DIR, r"validation_predictions")
    FIT_TOKENIZER_PATH = os.path.join(SQUAD_DATA_DIR, "fit_tokenizer.pickle")
    LOGS_DIR = r"./logs"
    TENSORBOARD_LOGS_DIR = os.path.join(LOGS_DIR, "tensorboard_logs")
    MODEL_CHECKPOINTS_DIR = os.path.join(LOGS_DIR, "model_checkpoints")
    MODEL_CHECKPOINT_FILE = os.path.join(MODEL_CHECKPOINTS_DIR, "model_{epoch:02d}.hdf5")
    METRICS_LOGS_FILE = os.path.join(LOGS_DIR, "metrics.csv")
    WITHIN_EPOCH_METRICS_FILE = os.path.join(LOGS_DIR, "within_epoch_metrics.csv")

    MAX_CONTEXT_LENGTH = 400
    MAX_QUESTION_LENGTH = 60
    MAX_ANSWER_LENGTH = 30
    VOCAB_SIZE = 30000
    EMBEDDING_DIM = 300
    D_MODEL = 128
    N_HEADS = 8
    N_EMB_ENC_BLOCKS = 1
    N_MOD_ENC_BLOCKS = 7
    ANS_NOT_FOUND = "ANSWER_NOT_FOUND" # Token for answer-span not found in tokenized context.
    SHUFFLE_DATA = True

    PRETRAINED_EMBEDDINGS_DIR = r"./data/glove"
    PRETRAINED_EMBEDDINGS_PATH = os.path.join(PRETRAINED_EMBEDDINGS_DIR, "glove.840B.{}d.txt".format(EMBEDDING_DIM))
    REDUCED_EMBEDDING_MATRIX_PATH = os.path.join(PRETRAINED_EMBEDDINGS_DIR, "reduced_embeddings.npy")

    VERY_NEGATIVE_NUMBER = -10**9

    # Training Configs:
    LIMIT_TRAIN = None
    LIMIT_VAL = None
    VALIDATE_ON_EPOCH_END = True
    BATCH_SIZE = 24
    BETWEEN_LAYER_DROPOUT_RATE = 0.1
    L2_LAMBDA = 3*(10**-7) # see section 4.1.1 in attention-question-answering paper.
    ADAM_b1 = 0.8
    ADAM_b2 = 0.999
    ADAM_eps = 10**-7
    GRAD_CLIP = 5 # not mentioned in paper, but applied in Google's implementation.
    LR_WARMUP_STEPS = 1000
    CONST_LR = 0.001
    INITIAL_LR = 0.0000001
    TRAINING_EPOCHS = 20
    CHECKPOINT_EVERY_N_EPOCHS = 1
    APPLY_EMA = True
    EMA_DECAY = 0.999
    CONV_ACTIVATION = "relu" # activation for the convs in the encoder blocks.
    RELU_INIT = "he_uniform"

    # Evaluation Configs:
    EVALUATE_WITHIN_EPOCH_EVERY_N_STEPS = 28839 // BATCH_SIZE # 3 evaluations per epoch
    WITHIN_EPOCH_EVALUATION_LIMIT = 3000 if LIMIT_VAL is None else min(LIMIT_VAL, 1000)
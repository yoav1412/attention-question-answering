import pickle
from layers import *
from keras.utils import to_categorical
from keras.optimizers import Adam
from model import build_model, get_squad_data_for_model, LearningRateCallback
from keras.utils import plot_model
from evaluation import EvaluationCallback
from keras.callbacks import TensorBoard
from parse_squad_data import SquadExample, Answer
import os
import tensorflow as tf
import numpy as np
from conf import Config as cf

tf.set_random_seed(1412)
np.random.seed(1412)

model = build_model(cf)
optimizer = Adam(beta_1=cf.ADAM_b1, beta_2=cf.ADAM_b2, epsilon=cf.ADAM_eps, clipnorm=cf.GRAD_CLIP)


model.compile(
            optimizer=optimizer,
            loss=['categorical_crossentropy', 'categorical_crossentropy', 'mae', 'mae'],
            loss_weights=[0.5, 0.5, 0, 0])

tokenized_contexts, tokenized_questions, y_start_positions, y_end_positions = \
    get_squad_data_for_model(cf.PARSED_TRAIN_DATA_PATH, shuffle_data=cf.SHUFFLE_DATA)

one_hot_start_positions = to_categorical(y_start_positions, num_classes=cf.MAX_CONTEXT_LENGTH)
one_hot_end_positions = to_categorical(y_end_positions, num_classes=cf.MAX_CONTEXT_LENGTH)



os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
plot_model(model, to_file='model.png')

eval_callback = EvaluationCallback()
lr_callback = LearningRateCallback()
tensorboard_callback = TensorBoard(log_dir=cf.TENSORBOARD_LOGS_DIR)


model.fit(x=[tokenized_contexts, tokenized_questions],
          y=[one_hot_start_positions, one_hot_end_positions, y_start_positions, y_end_positions],
          batch_size=cf.BATCH_SIZE, epochs=cf.TRAINING_EPOCHS,
          callbacks=[lr_callback,eval_callback])




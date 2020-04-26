from keras.layers import Layer
import tensorflow as tf
import numpy as np


class PositionalEncoding(Layer):
    def __init__(self, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)

    def build(self, input_shape):
        super(PositionalEncoding, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def positional_encoding(self, position, d_model):
        """
        Adapted from: https://www.tensorflow.org/tutorials/text/transformer#positional_encoding
        """
        def get_angles(pos, i, d_model):
            angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
            return pos * angle_rates

        angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                                np.arange(d_model)[np.newaxis, :],
                                d_model)

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        position, d = x.get_shape()[1:]
        position = int(position)
        d = int(d)
        pe = self.positional_encoding(position, d)
        return x + pe

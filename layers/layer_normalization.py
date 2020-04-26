from keras.layers import Layer
import tensorflow as tf
from conf import Config as cf
import keras.backend as K
from keras.regularizers import l2

class LayerNormalization(Layer):
    """
    Layer Normalization, presented in https://arxiv.org/pdf/1607.06450.pdf
    """
    def __init__(self, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.scale = self.add_weight(name='layer_norm_scale',
                                    shape=(input_shape[-1]),
                                    initializer=tf.ones_initializer(),
                                    regularizer = l2(cf.L2_LAMBDA),
                                    trainable=True)
        self.bias = self.add_weight(name='layer_norm_bias',
                                    shape=(input_shape[-1]),
                                    initializer=tf.zeros_initializer(), regularizer=l2(cf.L2_LAMBDA),
                                    trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x, mask=None, training=None):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
        norm_x = (x - mean) * tf.rsqrt(variance + K.epsilon())
        return norm_x * self.scale + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape

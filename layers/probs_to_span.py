from keras.layers import Layer, Lambda
import tensorflow as tf

class ProbsToSpan(Layer):
    def __init__(self, **kwargs):
        super(ProbsToSpan, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ProbsToSpan, self).build(input_shape)

    def call(self, probs):
        start_probs, end_probs = probs
        probs_matrix = Lambda(lambda z: tf.matmul(z[0], z[1], transpose_b=True))([start_probs, end_probs])
        upper_trig = Lambda(lambda mat: tf.matrix_band_part(mat, 0, -1))(probs_matrix) # Keep only cells where predicted start is before predicted end

        # upper_trig contains joing probabilities for (start,end) positions. We find (predicted_start, predicted_end) = argmax(upper_trig):
        predicted_start = Lambda(lambda mat: tf.argmax(tf.reduce_max(mat, axis=-1), axis=1))(upper_trig)
        predicted_end = Lambda(lambda mat: tf.argmax(tf.reduce_max(mat, axis=-2), axis=1))(upper_trig)

        predicted_start = Lambda(lambda z: tf.cast(z, tf.float32))(predicted_start)
        predicted_end = Lambda(lambda z: tf.cast(z, tf.float32))(predicted_end)

        return [predicted_start, predicted_end]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0][0],1), (input_shape[0][0],1)]
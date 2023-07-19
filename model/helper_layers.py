import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


class ShiftLayer(Layer):
    def build(self, input_shape):
        # prepare convolution filter that performs shifting
        # input is of shape [batch, 1, in_width, in_channels]
        # shiftFilter is of shape [1, filter_width, in_channels, 1]
        num_units = input_shape[2]
        shifted_maps = num_units // 3
        baseFilter = [[0, 1, 0], [1, 0, 0], [0, 0, 1]] * shifted_maps
        baseFilter = baseFilter + [[0, 1, 0]] * (num_units - shifted_maps*3)
        shiftFilter = tf.constant(np.transpose(baseFilter), dtype=self.compute_dtype)
        self.shift_filter = tf.expand_dims(tf.expand_dims(shiftFilter, 0), 3)
        return super().build(input_shape)

    def call(self, mem):
        mem_shifted = tf.squeeze(tf.nn.depthwise_conv2d(tf.expand_dims(mem, 1), self.shift_filter, [1, 1, 1, 1], 'SAME'), [1])
        return mem_shifted


def faro_shuffle(a):
    a1, a2 = tf.split(a, 2, axis=1)
    res = tf.concat([a1, a2], axis=-1)
    res = tf.reshape(res, a.shape)
    return res


def faro_shuffle_reverse(a):
    a1 = a[:,::2,:]
    a2 = a[:,1::2,:]
    res = tf.concat([a1, a2], axis=1)
    return res

def instance_norm(cur, axis=-1, epsilon=1e-5):
    """Normalize each element based on variance"""
    variance = tf.reduce_mean(tf.square(cur), axis, keepdims=True)
    cur = cur * tf.math.rsqrt(variance + epsilon)
    return cur

def gelu(x):
    return x * tf.sigmoid(1.702*x)

def inv_sigmoid(y):
    return np.log(y / (1 - y))

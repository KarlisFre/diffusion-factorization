import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D, Dropout
from tensorflow.keras.layers import Layer
from model.helper_layers import ShiftLayer

"""
Reimplementation of the Neural GPU from 'Improving the Neural GPU Architecture for Algorithm Learning' (https://arxiv.org/abs/1702.08727)
Code adapted from https://github.com/LUMII-Syslab/DNGPU
"""

n_log_images = 6

class GatedConvLayer(Layer):
    def __init__(self, kernel_width, droput_rate=0.0):
        super(GatedConvLayer, self).__init__()
        self.kernel_width = kernel_width
        self.drop = Dropout(droput_rate)
        self.shift_layer = ShiftLayer()
        self.lr_adjust = 10.

    def build(self, input_shape):
        num_units = input_shape[2]
        self.num_units = num_units
        self.conv_reset = Conv1D(num_units, self.kernel_width,
                                 padding='same', name='conv_reset', bias_initializer=tf.constant_initializer(0.5))
        self.conv_gate = Conv1D(num_units, self.kernel_width,
                                 padding='same', name='conv_gate', bias_initializer=tf.constant_initializer(0.7))
        self.conv_candidate = Conv1D(num_units, self.kernel_width,
                                 padding='same', name='conv_candidate', bias_initializer=tf.constant_initializer(0.0))

        return super().build(input_shape)

    def call(self, mem, training=False):
        mem_shifted = self.shift_layer(mem)
        reset = tf.sigmoid(self.conv_reset(mem))
        gate = tf.sigmoid(self.conv_gate(mem))
        candidate = tf.tanh(self.conv_candidate(reset * mem))
        candidate = self.drop(candidate)
        output = gate*mem_shifted + (1 - gate)*candidate

        return output, candidate, gate, reset


class DNGPU(Model):
    def __init__(self, dropout_rate = 0.1):
        super().__init__()
        self.compute_layer = GatedConvLayer(3, droput_rate=dropout_rate)

    def log_image(self, image, name):
        image *= tf.math.rsqrt(tf.reduce_mean(tf.square(image))+1e-6)
        tf.summary.image(name, tf.transpose(image * 0.25 + 0.5, [3, 0, 2, 1]), max_outputs=n_log_images)

    def call(self, x, training = False, log_in_tb = False):
        length = tf.shape(x)[1]
        step_mem = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=True)
        step_candidates = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=True)
        gates = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=True)
        gates1 = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=True)
        log_in_tb = log_in_tb and training and tf.summary.experimental.get_step()%100==0
        step_mem = step_mem.write(0, tf.cast(x[0:1, :, 0:n_log_images], tf.float32))

        for i in range(length):
            x, candidate, gate, res1 = self.compute_layer(x, training = training)
            step_mem = step_mem.write(i+1, tf.cast(x[0:1, :, 0:n_log_images], tf.float32))
            step_candidates = step_candidates.write(i, tf.cast(candidate[0:1, :, 0:n_log_images], tf.float32))
            gates = gates.write(i, tf.cast(gate[0:1, :, 0:n_log_images], tf.float32))
            gates1 = gates1.write(i, tf.cast(res1[0:1, :, 0:n_log_images], tf.float32))

        mem_all = step_mem.stack()
        step_candidates = step_candidates.stack()
        gates = gates.stack()
        gates1 = gates1.stack()

        if log_in_tb:
            self.log_image(mem_all, "mem")
            self.log_image(step_candidates, "cand")
            self.log_image(gates, "res_middle1")
            tf.summary.histogram("last_mem", x[0])
            tf.summary.histogram("res_middle", gates1)
        return x

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv1D, Dropout
from tensorflow.keras.layers import Layer
from model.helper_layers import ShiftLayer, faro_shuffle, faro_shuffle_reverse, inv_sigmoid, instance_norm, gelu

n_log_images = 6

#
# This file contains the proposed neural architecture for the denoising factorization task
#

# Convolutional Shuffle Unit
class CSU(Layer):
    def __init__(self, kernel_width, residual_weight=0.9, droput_rate=0.0):
        super(CSU, self).__init__()
        self.kernel_width = kernel_width
        self.residual_weight = residual_weight
        self.drop = Dropout(droput_rate)
        self.lr_adjust = 10.

    def build(self, input_shape):
        num_units = input_shape[2]
        self.num_units = num_units
        self.middle_units = num_units * 4
        self.conv1 = Conv1D(self.middle_units, self.kernel_width, padding='same', name='conv1')
        self.dense3 = Dense(num_units, name='dense3')

        self.residual_scale_param = self.add_weight(name="residual_scale", shape=[num_units],
                                                    initializer=tf.constant_initializer(inv_sigmoid(self.residual_weight) / self.lr_adjust))
        self.rezero = self.add_weight(name="rezero", shape=[num_units], initializer=tf.zeros_initializer())

        return super().build(input_shape)

    def call(self, mem, training=False):
        mem_drop = self.drop(mem, training=training)
        mem_shuffled = faro_shuffle(mem_drop)
        mem_shuffled1 = faro_shuffle_reverse(mem_drop)

        mem_drop = tf.concat([mem_drop, mem_shuffled, mem_shuffled1], axis=-1)
        res = self.conv1(mem_drop)
        res = instance_norm(res)
        res1 = res
        res = gelu(res)
        res_middle = res
        candidate = self.dense3(res)
        if training: candidate += tf.random.normal(tf.shape(candidate), stddev=0.001, dtype=self.compute_dtype)

        residual_scale = tf.sigmoid(self.residual_scale_param * self.lr_adjust)
        output = residual_scale * mem + candidate * self.rezero

        return output, candidate, res_middle, res1

# Recurrent Convolutional Shuffle Unit
class RCSU(Model):
    def __init__(self, dropout_rate = 0.1):
        super().__init__()
        estimated_depth = 20
        residual_weight = np.exp(np.log(0.1) / estimated_depth)
        self.compute_layer = CSU(3, residual_weight, droput_rate=dropout_rate)

    def log_image(self, image, name):
        image *= tf.math.rsqrt(tf.reduce_mean(tf.square(image))+1e-6)
        tf.summary.image(name, tf.transpose(image * 0.25 + 0.5, [3, 0, 2, 1]), max_outputs=n_log_images)

    def call(self, x, training = False, log_in_tb = False):
        length = tf.shape(x)[1]
        n_bits = tf.cast(tf.round(tf.math.log(tf.cast(length, tf.float32))/tf.math.log(2.)), tf.int32)
        step_mem = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=True)
        step_candidates = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=True)
        gates = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=True)
        gates1 = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=True)
        log_in_tb = log_in_tb and training and tf.summary.experimental.get_step()%100==0
        step_mem = step_mem.write(0, tf.cast(x[0:1, :, 0:n_log_images], tf.float32))

        depth = tf.maximum(length//2, n_bits*4)
        for i in range(depth):
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

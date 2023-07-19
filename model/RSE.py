import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer
from model.helper_layers import gelu

"""
    Implementation of Residual Shuffle Exchange network (https://arxiv.org/abs/2004.04662).
"""

class ShuffleLayer(Layer):
    def __init__(self, do_ror):
        self.do_ror = do_ror
        super().__init__()

    def build(self, input_shape):
        length = input_shape[1]
        rev_indices_rol = [(x * 2) % length for x in range((length + 1) // 2)] + [(x * 2 + 1) % length for x in range(length // 2)]

        if self.do_ror:
            rev_indices_ror = [-1 for _ in range(length)]
            for x in range(length): rev_indices_ror[rev_indices_rol[x]] = x
            self.rev_indices = rev_indices_ror
        else:
            self.rev_indices = rev_indices_rol

        return super().build(input_shape)

    def call(self, mem):
        mem_shuffled = tf.gather(mem, self.rev_indices, axis=1)
        return mem_shuffled

class LinearTransform(Layer):
    """
    Linear Transformation layer along feature dimension.
    """

    def __init__(self, name, n_out, bias_start=0.0, init_scale=1.0, add_bias=True, **kwargs):
        super(LinearTransform, self).__init__(trainable=True, name=name, **kwargs)
        self.n_out = n_out
        self.bias_start = bias_start
        self.init_scale = init_scale
        self.kernel = None
        self.bias_term = None
        self.n_in = None
        self.add_bias = add_bias

    def build(self, input_shape: tf.TensorShape):
        self.n_in = input_shape.as_list()[-1]

        initializer = tf.keras.initializers.VarianceScaling(scale=self.init_scale, mode="fan_avg", distribution="uniform")
        self.kernel = self.add_weight("CvK", [self.n_in, self.n_out], initializer=initializer)

        if self.add_bias:
            self.bias_term = self.add_weight("CvB", [self.n_out], initializer=tf.constant_initializer(self.bias_start))

    def call(self, inputs, **kwargs):

        input_shape = inputs.get_shape().as_list()

        in_shape = 1
        for shape in input_shape[:-1]:
            in_shape *= shape

        reshape_in = [in_shape, self.n_in]
        reshape_out = [shape for shape in input_shape[:-1]] + [self.n_out]

        res = tf.matmul(tf.reshape(inputs, reshape_in), self.kernel)
        res = tf.reshape(res, reshape_out)

        if self.add_bias:
            res = res + self.bias_term

        return res


class SwitchLayer(Layer):
    """
    Switch Layer with Residual Switch Units that have 2 inputs and 2 outputs each.
    Residual Switch Unit: (https://arxiv.org/abs/2004.04662).
    """

    def __init__(self, name, channel_count=2, dropout_rate=0.1, **kwargs):
        super(SwitchLayer, self).__init__(name=name, **kwargs)
        self.channel_count = channel_count
        self.dropout_rate = dropout_rate
        self.residual_weight = 0.9
        self.candidate_weight = np.sqrt(1 - self.residual_weight ** 2) * 0.25
        self.scale_init = np.log(self.residual_weight / (1 - self.residual_weight))

        self.num_units = None
        self.reshaped_units = None
        self.residual_scale = None
        self.layer_norm = None
        self.dropout = None

        self.linear_one = None
        self.linear_two = None

    def build(self, input_shape):
        self.num_units = input_shape.as_list()[2]
        self.reshaped_units = self.num_units * self.channel_count

        initializer = tf.constant_initializer(self.scale_init)
        self.residual_scale = self.add_weight("residual", [self.reshaped_units], initializer=initializer)

        self.linear_one = LinearTransform("linear_one", self.reshaped_units * 2, add_bias=False)
        self.linear_two = LinearTransform("linear_two", self.reshaped_units)

        self.layer_norm = LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, inputs, training=False, **kwargs):
        batch_size, length, num_units = inputs.shape.as_list()[:3]
        inputs = tf.reshape(inputs, [batch_size, length // self.channel_count, self.reshaped_units])
        dropout = self.dropout(inputs, training=training)

        first_linear = self.linear_one(dropout)
        norm = self.layer_norm(first_linear)
        second_linear = self.linear_two(gelu(norm))

        residual_scale = tf.nn.sigmoid(self.residual_scale)

        candidate = residual_scale * inputs + second_linear * self.candidate_weight
        return tf.reshape(candidate, [batch_size, length, self.num_units])


class LayerNormalization(Layer):

    def __init__(self, axis=1, epsilon=1e-6, **kwargs):
        self.axis = axis
        self.epsilon = epsilon
        self.bias = None
        super(LayerNormalization, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        variance = tf.reduce_mean(tf.square(inputs), self.axis, keepdims=True)
        return inputs * tf.math.rsqrt(variance + self.epsilon)


class RSE(Layer):
    """Implementation of Residual Shuffle Exchange network
    This implementation expects 3-D inputs - [batch_size, length, channels]
    Output shape will be same as input shape
    RSE output is output from the last SwitchLayer layer. No additional output processing is applied.
    """

    def __init__(self, block_count=1, **kwargs):
        """
        :param block_count: Determines Beneš block count that are chained together
        """
        super().__init__(**kwargs)
        self.block_count = block_count
        self.block_layers = None
        self.last_switch_layer = None

    def build(self, input_shape):

        self.block_layers = [0]*self.block_count
        for i in range(self.block_count):
            self.block_layers[i] = {
                "forward": SwitchLayer("forward"),
                "reverse": SwitchLayer("reverse")
            }

        self.last_switch_layer = SwitchLayer("last")

    def log_image(self, image, name):
        image *= tf.math.rsqrt(tf.reduce_mean(tf.square(image))+1e-6)
        tf.summary.image(name, tf.transpose(image * 0.25 + 0.5, [3, 0, 2, 1]), max_outputs=6)

    def call(self, inputs, training=False, **kwargs):
        input_shape = inputs.get_shape().as_list()
        level_count = (input_shape[1] - 1).bit_length() - 1
        current_layer = inputs

        step_mem = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=True)
        i=0
        n_log_images = 6

        for block_nr in range(self.block_count):

            with tf.name_scope(f"benes_block_{block_nr}"):
                for _ in range(level_count):
                    current_layer = self.block_layers[block_nr]["forward"](current_layer, training=training)
                    step_mem = step_mem.write(i, tf.cast(current_layer[0:1, :, 0:n_log_images], tf.float32))
                    i+=1
                    current_layer = ShuffleLayer(do_ror=False)(current_layer)

                for _ in range(level_count):
                    current_layer = self.block_layers[block_nr]["reverse"](current_layer, training=training)
                    step_mem = step_mem.write(i, tf.cast(current_layer[0:1, :, 0:n_log_images], tf.float32))
                    i+=1

                    current_layer = ShuffleLayer(do_ror=True)(current_layer)

        current_layer = self.last_switch_layer(current_layer, training=training)
        step_mem = step_mem.write(i, tf.cast(current_layer[0:1, :, 0:n_log_images], tf.float32))
        mem_all = step_mem.stack()
        log_in_tb = kwargs["log_in_tb"]

        if log_in_tb:
            self.log_image(mem_all, "mem")
        return current_layer

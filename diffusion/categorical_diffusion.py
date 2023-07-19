import tensorflow as tf
import numpy as np
import config

class CategoricalDiffusion():

    def __init__(self, n_classes = 2) -> None:
        super().__init__()
        self.n_classes = n_classes

    def distribution_at_time(self, x, time_increment):
        return x*(1-time_increment)+time_increment/self.n_classes


    def randomized_rounding(self, x):
        # # sample from categorical distribution
        shape = list(x.shape)
        shape[-1] = 1
        cum_prob = np.cumsum(x, axis=-1)
        r = np.random.uniform(size=shape)
        # argmax finds the index of the first True value in the last axis.
        res = np.argmax(cum_prob > r, axis=-1)
        res = tf.one_hot(res, self.n_classes)
        return res

    # performs one distribution step in reverse
    # x is the current distribution
    # x0 is the predicted value at t=0
    # t_increment = 1/N_steps
    # this corresponds to how it should be in the theory
    def reverse_distribution_step_thoeretic(self, x, x0, t, t_increment):
        x_new = self.distribution_at_time(x0, t)
        alpha_t = (1-t)/(1-t+t_increment)
        x_unnormed = self.distribution_at_time(x, 1-alpha_t) * x_new
        x = x_unnormed / (tf.reduce_sum(x_unnormed, axis=-1, keepdims=True) + 1e-8)
        return x

    # performs one distribution step in reverse, our version, tailored to factorization
    # x is the current distribution
    # x0 is the predicted value at t=0
    # t_increment = 1/N_steps
    def reverse_distribution_step(self, x, x0, t, t_increment):
        x_new = self.distribution_at_time(x0, t)
        step_len = 0.1
        x = x * (1 - step_len) + x_new * step_len
        return x

    # this method should not use tf
    def random_at_t(self, both_nums_hot_clean, length, noise_scale):
        distribution = self.distribution_at_time(both_nums_hot_clean, noise_scale)
        return self.randomized_rounding(distribution)


    # The train loss is KL divergence of labels and predictions both transferred to time t
    # note that t is array for each sample
    def train_loss(self, labels, prediction_logits, t):
        t = t[..., tf.newaxis, tf.newaxis]
        labels_at_t = self.distribution_at_time(labels, tf.minimum(t+config.label_smoothing,1))
        probs = tf.nn.softmax(prediction_logits, axis=-1) #maybe use logscale
        probs_at_t = self.distribution_at_time(probs, t)
        KL = -tf.reduce_sum(labels_at_t*tf.math.log(tf.maximum(probs_at_t, 1e-20)), axis=-1)+\
               tf.reduce_sum(labels_at_t*tf.math.log(tf.maximum(labels_at_t, 1e-20)), axis=-1)
        item_loss = tf.reduce_mean(KL, axis=1)
        return 100*tf.reduce_mean(item_loss)

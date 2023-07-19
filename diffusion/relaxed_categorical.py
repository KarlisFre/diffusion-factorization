import tensorflow as tf
import numpy as np

from utils import sample_gumbel, sample_gumbel_tf
from diffusion.categorical_diffusion import CategoricalDiffusion
from scipy.special import softmax

class RelaxedCategoricalDiffusion(CategoricalDiffusion):

    def randomized_rounding(self, x):
        log_probs = np.log(np.maximum(x, 1e-20))
        log_probs += sample_gumbel(x.shape)
        res = softmax(log_probs, axis=-1)
        return res


    def randomized_rounding_tf(self, x):
        log_probs = tf.math.log(tf.maximum(x, 1e-20))
        log_probs += sample_gumbel_tf(x.shape)
        res = tf.nn.softmax(log_probs, axis=-1)
        return res

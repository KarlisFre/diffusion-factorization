import numpy as np
import tensorflow as tf

def one_hot(a, num_classes):
  a = np.array(a)
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def sigmoid(x):
  return np.where(x >= 0,
                  1 / (1 + np.exp(-x)),
                  np.exp(x) / (1 + np.exp(x)))

def inv_sigmoid(y, eps=1e-10):
  y = np.maximum(y, eps)
  y = np.minimum(y, 1 - eps)
  return np.log(y / (1 - y))

def inv_sigmoid_tf(y, eps=1e-10):
  y = tf.maximum(y, eps)
  y = tf.minimum(y, 1 - eps)
  return tf.math.log(y / (1 - y))

def sample_logistic(shape, eps=1e-20):
  U = np.random.uniform(size=shape, low=eps, high=1 - eps)
  return np.log(U / (1 - U))

def sample_logistic_tf(shape, eps=1e-20):
  U = tf.random.uniform(shape, minval=eps, maxval=1 - eps)
  return tf.math.log(U / (1 - U))

def sample_gumbel(shape, eps=1e-20):
  U = np.random.uniform(size=shape)
  return -np.log(-np.log(U + eps) + eps)

def sample_gumbel_tf(shape, eps=1e-20):
  U = tf.random.uniform(shape)
  return -tf.math.log(-tf.math.log(U + eps) + eps)

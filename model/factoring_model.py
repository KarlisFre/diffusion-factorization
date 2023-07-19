import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout

from model.rcsu_model import RCSU
from model.helper_layers import instance_norm, gelu


class FactoringModel(Model):
  def __init__(self, in_maps, hidden_maps, n_classes, dropout_rate = 0.0):
    super(FactoringModel, self).__init__()
    self.d1 = Dense(hidden_maps, name = 'input1')
    self.d1a = Dense(hidden_maps, name='input2')
    self.d2 = Dense(n_classes, dtype=tf.float32, name = 'output')
    self.rgpu = RCSU(dropout_rate=dropout_rate)
    self.drop = Dropout(dropout_rate)

  def call(self, x, training, log_in_tb = False):
    x = self.d1(x, training = training)
    x = instance_norm(x)
    x = gelu(x)
    x = self.d1a(x)*0.25
    x = self.rgpu(x, training = training, log_in_tb = log_in_tb)
    x = self.drop(x, training = training)
    x = self.d2(x, training = training)

    return x

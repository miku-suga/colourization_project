import tensorflow as tf
import numpy as np
from tensorflow.keras import Model

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        pass

    def call(self, inputs, is_testing=False):
        pass

    def loss(self, logits, labels):
        pass
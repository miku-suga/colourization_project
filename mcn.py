import tensorflow as tf
import numpy as np
from tensorflow.keras import Model

class MainColorizationNet(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        pass

    def call(self, inputs, is_testing=False):
        pass

    def loss(self, logits, labels):
        pass

    def accuracy(self, logits, labels):
        pass
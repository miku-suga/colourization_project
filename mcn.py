import tensorflow as tf
import numpy as np
from tensorflow.keras import Model

""" 
    MCN Encoder 
    Input: luminance channel of input image and reference image
    Output: Feature matrix of input image and reference image
            lumniance channel of input image
"""
class Encoder(tf.keras.Model):
    def __init__(self, r_l, t_l, m_1, m_2, m_3):
        super(Model, self).__init__()
        """ Init input matrices """
        self.r_l = r_l
        self.t_l = t_l

        """ Init layers """

    def call(self, inputs, is_testing=False):
        pass

""" 
    MCN Decoder 1
    Input: fused color matrix m_1,
           lumninace channel of input image
    Output: ab channel matrix of the input image 
            luminance channel of input image
"""
class Decoder1(tf.keras.Model):
    def __init__(self, t_l, m_1):
        super(Model, self).__init__()
        """ Init input matrices """
        self.t_l = t_l
        self.m_1 = m_1

        """ Init layers """

    def call(self, inputs, is_testing=False):
        pass

""" 
    MCN Decoder 2
    Input: fused color matrix m_2,
           lumninace channel of input image
    Output: ab channel matrix of the input image 
            luminance channel of input image
"""
class Decoder2(tf.keras.Model):
    def __init__(self, t_l, m_2):
        super(Model, self).__init__()
        """ Init input matrices """
        self.t_l = t_l
        self.m_2 = m_2

        """ Init layers """

    def call(self, inputs, is_testing=False):
        pass

""" 
    MCN Decoder 3
    Input: fused color matrix m_3,
           lumninace channel of input image
    Output: ab channel matrix of the input image 
            luminance channel of input image
"""
class Decoder3(tf.keras.Model):
    def __init__(self, t_l, m_3):
        super(Model, self).__init__()
        """ Init input matrices """
        self.t_l = t_l
        self.m_3 = m_3

        """ Init layers """

    def call(self, inputs, is_testing=False):
        pass


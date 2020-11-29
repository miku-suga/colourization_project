import tensorflow as tf
import numpy as np
from tensorflow.keras import Model

""" Make Resblock Class hers """

""" 
    MCN Encoder 
    Input: luminance channel of input image and reference image
    Output: Feature matrix of input image and reference image
            lumniance channel of input image
"""
class Encoder(tf.keras.Model):
    def __init__(self, r_l, t_l, m_1, m_2, m_3):
        super(Encoder, self).__init__()
        """ Init input matrices """
        self.r_l = r_l
        self.t_l = t_l

        """ Init layers """
        self.conv_1_1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')
        self.conv_1_2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()

        self.conv_2_1 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')
        self.conv_2_2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()

        self.conv_3_1 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')
        self.conv_3_2 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')
        self.conv_3_3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')
        self.batch_norm_3 = tf.keras.layers.BatchNormalization()

        self.conv_4_1 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')
        self.conv_4_2 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')
        self.conv_4_3 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')
        self.batch_norm_4 = tf.keras.layers.BatchNormalization()

        """ add dilation 1 """

        """ add dilation 2 """
        


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
        super(Decoder1, self).__init__()
        """ Init input matrices """
        self.t_l = t_l
        self.m_1 = m_1

        """ Init layers """
        self.resconv_1 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        """ add 2 resblocks """
        """ conv2dtranspose here """

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
        super(Decoder2, self).__init__()
        """ Init input matrices """
        self.t_l = t_l
        self.m_2 = m_2

        """ Init layers """
        self.resconv_1 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        """ add 2 resblocks """
        """ conv2dtranspose here """

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
        super(Decoder3, self).__init__()
        """ Init input matrices """
        self.t_l = t_l
        self.m_3 = m_3

        """ Init layers """
        self.resconv_1 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        """ add 2 resblocks """
        """ conv2dtranspose here """

    def call(self, inputs, is_testing=False):
        pass


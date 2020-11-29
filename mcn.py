import tensorflow as tf
import numpy as np
from tensorflow.keras import Model

""" 
    ResBlock
    Input: some input x
    Output: residual of x
"""
class ResBlock(tf.keras.Model):
    def __init__(self, dim):
        super(ResBlock, self).__init__()

        """ Init layers """
        self.conv_1 = tf.keras.layers.Conv2D(dim, 3, activation='relu', padding='same')
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.conv_2 = tf.keras.layers.Conv2D(dim, 3, activation='relu', padding='same')
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()

    def call(self, input):
        return input + self.batch_norm_2(self.conv_2(self.batch_norm_1(self.conv_1(input))))


""" 
    MCN Encoder 
    Input: luminance channel of input image and reference image
    Output: Feature matrix of input image and reference image
            lumniance channel of input image
"""
class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()

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
        


    def call(self, r_l, t_l, m_1, m_2, m_3, is_testing=False):
        pass

""" 
    MCN Decoder 1
    Input: fused color matrix m_1,
           lumninace channel of input image
    Output: ab channel matrix of the input image 
            luminance channel of input image
"""
class Decoder1(tf.keras.Model):
    def __init__(self):
        super(Decoder1, self).__init__()

        """ Init layers """
        self.resconv_1 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.deconv_up = tf.keras.layers.Conv2DTranspose(256, 4, strides=2, padding='same')
        self.deconv_short = tf.keras.layers.Conv2D(256, 3, padding='same') 
        self.relu_1 = tf.keras.layers.ReLU()
        self.deconv_1 = tf.keras.layers.Conv2D(256, 3, padding='same')
        self.relu_2 = tf.keras.layers.ReLU()
        self.deconv_2 = tf.keras.layers.Conv2D(256, 3, padding='same')
        self.relu_3 = tf.keras.layers.ReLU()
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()

    def call(self, t_l, m_1, is_testing=False):
        pass

""" 
    MCN Decoder 2
    Input: fused color matrix m_2,
           lumninace channel of input image
    Output: ab channel matrix of the input image 
            luminance channel of input image
"""
class Decoder2(tf.keras.Model):
    def __init__(self):
        super(Decoder2, self).__init__()

        """ Init layers """
        self.resconv_1 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.deconv_up = tf.keras.layers.Conv2DTranspose(128, 4, strides=2, padding='same')
        self.deconv_short = tf.keras.layers.Conv2D(128, 3, padding='same') 
        self.relu_1 = tf.keras.layers.ReLU()
        self.deconv = tf.keras.layers.Conv2D(128, 3, padding='same') 
        self.relu_2 = tf.keras.layers.ReLU()
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()

    def call(self, t_l, m_2, is_testing=False):
        pass

""" 
    MCN Decoder 3
    Input: fused color matrix m_3,
           lumninace channel of input image
    Output: ab channel matrix of the input image 
            luminance channel of input image
"""
class Decoder3(tf.keras.Model):
    def __init__(self):
        super(Decoder3, self).__init__()

        """ Init layers """
        self.resconv_1 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.deconv_up = tf.keras.layers.Conv2DTranspose(128, 4, strides=2, padding='same')
        self.deconv_short = tf.keras.layers.Conv2D(128, 3, padding='same') 
        self.relu = tf.keras.layers.ReLU()
        self.deconv = tf.keras.layers.Conv2D(128, 3, padding='same') 
        self.leaky_relu = tf.keras.layers.LeakyReLU()

    def call(self, t_l, m_3, is_testing=False):
        pass


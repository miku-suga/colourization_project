import tensorflow as tf
import numpy as np
from tensorflow.keras import Model


""" 
    Gated Fusion Module
    Input: correlation matrix C
           6 feature matrices of 3 different sizes
    Output: 3 fused color features of 3 different sizes
"""
class GatedFusionModule(tf.keras.Model):
    def __init__(self):
        super(GatedFusionModule, self).__init__()

        """ Init layers """

    def call(self, corr, f_s_1, f_s_2, f_s_3, f_rh_1, f_rh_2, f_rh_3, is_testing=False):
        pass


""" 
    Semantic Assingment Module
    Input: features of input monochrome image, 
           features of luminance channel of reference image,
           ab channels of reference image
    Output: input class label G,
            correlation matrix C,
            3 color feature matrices of different sizes
"""
class SemanticAssignmentModule(tf.keras.Model):
    def __init__(self, height, width):
        super(SemanticAssignmentModule, self).__init__()
        
        self.height = height
        self.width = width

        self.kernel_size = 3
        
        self.rab_conv_1 = tf.keras.layers.Conv2D(64, self.kernel_size, activation='relu', padding='same')
        self.rab_conv_2 = tf.keras.layers.Conv2D(128, self.kernel_size, strides=(2, 2), activation='relu', padding='same')

        self.fa_conv_1 = tf.keras.layers.Conv2D(128, self.kernel_size, strides=(2, 2), activation='relu', padding='same')
        self.fa_conv_2 = tf.keras.layers.Conv2D(256, self.kernel_size, strides=(2, 2), activation='relu', padding='same')
        self.fa_conv_3 = tf.keras.layers.Conv2D(512, self.kernel_size, strides=(2, 2), activation='relu', padding='same')


    def call(self, t_lum, r_lum, r_ab, is_testing=False):
        pass

""" 
    Color Distribution Module
    Input: histogram of reference image
    Output: 3 features matrices of R_h of different sizes
            *these matrices are pre-spacial replication
"""
class ColorDistributionModule(tf.keras.Model):
    def __init__(self):
        super(ColorDistributionModule, self).__init__()

        self.kernel_size = 1

        """ Init layers """
        self.conv_1_1 = tf.keras.layers.Conv2D(512, self.kernel_size, activation='relu', padding='same')
        self.conv_1_2 = tf.keras.layers.Conv2D(512, self.kernel_size, activation='relu', padding='same')
        self.conv_1_3 = tf.keras.layers.Conv2D(512, self.kernel_size, activation='relu', padding='same')

        self.conv_2_1 = tf.keras.layers.Conv2D(512, self.kernel_size, activation='relu', padding='same')
        self.conv_2_2 = tf.keras.layers.Conv2D(256, self.kernel_size, activation='relu', padding='same')
        self.conv_2_3 = tf.keras.layers.Conv2D(128, self.kernel_size, activation='relu', padding='same')

    def call(self, r_hist, is_testing=False):
        conv_output = self.conv_1_3(self.conv_1_2(self.conv_1_1(r_hist)))
        f_1 = self.conv_2_1(conv_output)
        f_2 = self.conv_2_2(conv_output)
        f_3 = self.conv_2_3(conv_output)
        return f_1, f_2, f_3

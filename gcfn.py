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
    def __init__(self, corr, f_s_1, f_s_2, f_s_3, f_rh_1, f_rh_2, f_rh_3):
        super(Model, self).__init__()
        """ Init input matrices """
        self.corr = corr
        self.f_s_1 = f_s_1
        self.f_s_2 = f_s_2
        self.f_s_3 = f_s_3
        self.f_rh_1 = f_rh_1
        self.f_rh_2 = f_rh_2
        self.f_rh_3 = f_rh_3

        """ Init layers """

    def call(self, inputs, is_testing=False):
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
    def __init__(self, t_lum, r_lum, r_ab):
        super(Model, self).__init__()
        """ Init input matrices """
        self.t_lum = t_lum
        self.r_lum = r_lum
        self.r_ab = r_ab

        """ Init layers """

    def call(self, inputs, is_testing=False):
        pass



""" 
    Color Distribution Module
    Input: histogram of reference image
    Output: 3 features matrices of R_h of different sizes
"""
class ColorDistributionModule(tf.keras.Model):
    def __init__(self, r_hist):
        super(Model, self).__init__()
        """ Init input matrices """
        self.r_hist = r_hist

        """ Init layers """

    def call(self, inputs, is_testing=False):
        pass


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
        self.kernel_size = 3

        self.C_conv_1 = tf.keras.layers.Conv2D(64, self.kernel_size, strides=(2, 2), activation='relu', padding='same')
        self.C_conv_2 = tf.keras.layers.Conv2D(128, self.kernel_size, strides=(2, 2), activation='relu', padding='same')

    def call(self, corr, f_s_1, f_s_2, f_s_3, f_rh_1, f_rh_2, f_rh_3, is_testing=False):
        corr = self.C_conv_1(corr)
        corr = self.C_conv_2(corr)


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
    def __init__(self, num_classes):
        super(SemanticAssignmentModule, self).__init__()
        
        """ self.height = height """
        """ self.width = width """

        self.kernel_size = 3
        self.num_classes = num_classes

        self.rab_conv_1 = tf.keras.layers.Conv2D(64, self.kernel_size, activation='relu', padding='same')
        self.rab_conv_2 = tf.keras.layers.Conv2D(128, self.kernel_size, strides=(2, 2), activation='relu', padding='same')

        self.fa_conv_1 = tf.keras.layers.Conv2D(128, self.kernel_size, strides=(2, 2), activation='relu', padding='same')
        self.fa_conv_2 = tf.keras.layers.Conv2D(256, self.kernel_size, strides=(2, 2), activation='relu', padding='same')
        self.fa_conv_3 = tf.keras.layers.Conv2D(512, self.kernel_size, strides=(2, 2), activation='relu', padding='same')

        self.gtl_dense_1 = tf.keras.layers.Dense(1024, activation='relu')
        self.gtl_dense_2 = tf.keras.layers.Dense(1024, activation='relu')
        self.gtl_dense_3 = tf.keras.layers.Dense(num_classes, activation='softmax')


    def call(self, t_lum, r_lum, r_ab, is_testing=False):
        """ takes in r, t, output of encoder, r_ab and spit out correlation matrix features conf_1,2,3, class output G, and f_s1,2,3 """
        C = self.correlate(t_lum, r_lum)
        
        f_rab = self.rab_conv_1(r_ab)
        f_rab = self.rab_conv_2(f_rab)

        f_a = self.attention(C, f_rab)
        f_s3 = self.fa_conv_1(f_a)
        f_s2 = self.fa_conv_2(f_s3)
        f_s1 = self.fa_conv_3(f_s2)

        mp = tf.nn.max_pool(t_lum[3], 2, strides=2, padding='same')

        g_tl = self.gtl_dense_1(tf.reshape(mp, [len(mp), -1]))
        g_tl = self.gtl_dense_2(g_tl)
        g_tl = self.gtl_dense_3(g_tl)

        return (g_tl, C, f_s3, f_s2, f_s1)

    def correlate(self, t_lum, r_lum):
        # TODO: optimize this
        C = tf.zeros(len(t_lum), len(r_lum))
        for i in range(len(t_lum)):
            for j in range(len(r_lum)):
                norm_fac = tf.norm(t_lum[i]) * tf.norm(r_lum[j])
                dot_prod = tf.tensordot(t_lum[i], r_lum[j], axis=0)
                C[i, j] = norm_fac / dot_prod
        return C

    def attention(self, C, f_rab):
        # TODO: optimize this with tensordot
        exp_c = tf.exp(C)
        aij = exp_c / tf.reduce_sum(exp_c, axis=1)
        
        f_a = tf.zeros_like(f_rab)
        for i in range(len(aij)):
            f_a[i] = tf.sum(aij[i] * f_rab, axis=0)
        
        return f_a


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

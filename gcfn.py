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
    def __init__(self, img_height, img_width):
        super(GatedFusionModule, self).__init__()

        """ Init layers """
        self.kernel_size = 3

        self.height = img_height
        self.width = img_width

        self.C_conv_1 = tf.keras.layers.Conv2D(
            64, self.kernel_size, strides=(2, 2), activation='relu', padding='same')
        self.C_conv_2 = tf.keras.layers.Conv2D(
            128, self.kernel_size, strides=(2, 2), activation='relu', padding='same')

    def call(self, corr, align_1, align_2, align_3, global_1, global_2, global_3, is_testing=False):
        conf = self.C_conv_1(corr)
        assert conf.shape[1:] == (self.height*self.width // 4, 1024, 1)
        conf = self.C_conv_2(corr)
        assert conf.shape[1:] == (self.height*self.width // 4, 1, 1)


        conf_1 = tf.reshape(conf, [-1, self.height, self.width, 1])
        conf_2 = conf_1[:, :, ::2, ::2]
        conf_3 = conf_2[:, :, ::2, ::2]

        assert conf_1 == (self.height, self.width, 1)
        assert conf_2 == (self.height // 2, self.width // 2, 1)
        assert conf_3 == (self.height // 2, self.width // 2, 1)

        global_1 = tf.broadcast_to(global_1, align_1.shape)
        global_2 = tf.broadcast_to(global_2, align_2.shape)
        global_3 = tf.broadcast_to(global_3, align_3.shape)

        M3 = align_3 * conf_3 + global_1 * (1 - conf_3)
        M2 = align_2 * conf_2 + global_2 * (1 - conf_2)
        M1 = align_1 * conf_1 + global_3 * (1 - conf_1)

        assert M1 == (self.height // 8, self.width // 8, 512)
        assert M2 == (self.height // 4, self.width // 4, 256)
        assert M3 == (self.height // 2, self.width // 2, 128)
        
        return M1, M2, M3


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
    def __init__(self, num_classes, img_height, img_width):
        super(SemanticAssignmentModule, self).__init__()

        self.kernel_size = 3
        self.num_classes = num_classes

        self.height = img_height
        self.width = img_width

        self.rab_conv_1 = tf.keras.layers.Conv2D(
            64, self.kernel_size, activation='relu', padding='same')
        self.rab_conv_2 = tf.keras.layers.Conv2D(
            128, self.kernel_size, strides=(2, 2), activation='relu', padding='same')

        self.fa_conv_1 = tf.keras.layers.Conv2D(
            128, self.kernel_size, strides=(2, 2), activation='relu', padding='same')
        self.fa_conv_2 = tf.keras.layers.Conv2D(
            256, self.kernel_size, strides=(2, 2), activation='relu', padding='same')
        self.fa_conv_3 = tf.keras.layers.Conv2D(
            512, self.kernel_size, strides=(2, 2), activation='relu', padding='same')

        self.gtl_dense_1 = tf.keras.layers.Dense(512, activation='relu')
        self.gtl_dense_2 = tf.keras.layers.Dense(
            num_classes, activation='softmax')

    def call(self, t_lum, r_lum, r_ab, is_testing=False):
        """ takes in r, t, output of encoder, r_ab and spit out correlation matrix features conf_1,2,3, class output G, and f_s1,2,3 """

        assert t_lum.shape[1:] == (self.height // 2, self.width // 2, 1984)
        assert r_lum.shape[1:] == (self.height // 2, self.width // 2, 1984)
        
        C = self.correlate(t_lum, r_lum)
        assert C.shape[1:] == (self.height // 4, self.width // 4, 1)

        f_rab = self.rab_conv_1(r_ab)
        assert f_rab.shape[1:] == (self.height, self.width, 64)
        f_rab = self.rab_conv_2(f_rab)
        assert f_rab.shape[1:] == (self.height // 2, self.width // 2, 128)

        f_a = self.attention(C, f_rab)
        assert f_a.shape[1:] == (self.height, self.width, 128)
        f_s3 = self.fa_conv_1(f_a)
        assert f_s3.shape[1:] == (self.height // 2, self.width // 2, 128)
        f_s2 = self.fa_conv_2(f_s3)
        assert f_s2.shape[1:] == (self.height // 4, self.width // 4, 256)
        f_s1 = self.fa_conv_3(f_s2)
        assert f_s1.shape[1:] == (self.height // 8, self.width // 8, 512)

        mp = tf.nn.max_pool2d(t_lum[3], t_lum.shape,
                              strides=1, padding="VALID")
        assert mp.shape[1:] == (1, 1, 512)

        g_tl = self.gtl_dense_1(tf.reshape(mp, [len(mp), -1]))
        assert g_tl.shape[1:] == (512)
        g_tl = self.gtl_dense_2(g_tl)

        return (g_tl, C, f_s1, f_s2, f_s3)

    def correlate(self, t_lum, r_lum):
        # TODO: optimize this
        C = tf.zeros(len(t_lum), len(r_lum))
        for i in range(len(t_lum)):
            for j in range(len(r_lum)):
                norm_fac = tf.norm(t_lum[i]) * tf.norm(r_lum[j])
                dot_prod = tf.tensordot(t_lum[i], r_lum[j], axes=0)
                C[i, j] = norm_fac / dot_prod
        return C

    def attention(self, C, f_rab):
        # TODO: optimize this with tensordot
        exp_c = tf.exp(C)
        aij = exp_c / tf.reduce_sum(exp_c, axis=1)

        f_a = tf.zeros_like(f_rab)
        for i in range(len(aij)):
            f_a[i] = tf.reduce_sum(aij[i] * f_rab, axis=0)

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
        self.conv_1_1 = tf.keras.layers.Conv2D(
            512, self.kernel_size, activation='relu', padding='same')
        self.conv_1_2 = tf.keras.layers.Conv2D(
            512, self.kernel_size, activation='relu', padding='same')
        self.conv_1_3 = tf.keras.layers.Conv2D(
            512, self.kernel_size, activation='relu', padding='same')

        self.conv_2_1 = tf.keras.layers.Conv2D(
            512, self.kernel_size, activation='relu', padding='same')
        self.conv_2_2 = tf.keras.layers.Conv2D(
            256, self.kernel_size, activation='relu', padding='same')
        self.conv_2_3 = tf.keras.layers.Conv2D(
            128, self.kernel_size, activation='relu', padding='same')

    def call(self, r_hist, is_testing=False):
        conv_output = self.conv_1_3(self.conv_1_2(self.conv_1_1(r_hist)))
        assert conv_output.shape[1:] == (1, 1, 512)
        f_1 = self.conv_2_1(conv_output)
        assert f_1.shape[1:] == (1, 1, 512)
        f_2 = self.conv_2_2(conv_output)
        assert f_2.shape[1:] == (1, 1, 256)
        f_3 = self.conv_2_3(conv_output)
        assert f_3.shape[1:] == (1, 1, 128)
        return f_1, f_2, f_3

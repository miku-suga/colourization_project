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
        self.kernel_size = 1

        self.height = img_height
        self.width = img_width

    @tf.function
    def call(self, conf, align_1, align_2, align_3, global_1, global_2, global_3, is_testing=False):
        conf_1 = tf.reshape(
            conf, [conf.shape[0], self.height // 2, self.width // 2, 1])
        conf_2 = conf_1[:, ::2, ::2, :]
        conf_3 = conf_2[:, ::2, ::2, :]

        assert conf_1.shape[1:] == (self.height // 2, self.width // 2, 1)
        assert conf_2.shape[1:] == (self.height // 4, self.width // 4, 1)
        assert conf_3.shape[1:] == (self.height // 8, self.width // 8, 1)

        global_1 = tf.broadcast_to(global_1, align_3.shape)
        global_2 = tf.broadcast_to(global_2, align_2.shape)
        global_3 = tf.broadcast_to(global_3, align_1.shape)

        M1 = align_3 * conf_3 + global_1 * (1 - conf_3)
        M2 = align_2 * conf_2 + global_2 * (1 - conf_2)
        M3 = align_1 * conf_1 + global_3 * (1 - conf_1)

        assert M1.shape[1:] == (self.height // 8, self.width // 8, 512)
        assert M2.shape[1:] == (self.height // 4, self.width // 4, 256)
        assert M3.shape[1:] == (self.height // 2, self.width // 2, 128)

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
    def __init__(self, img_height, img_width):
        super(SemanticAssignmentModule, self).__init__()

        self.kernel_size = 3

        self.height = img_height
        self.width = img_width

        self.batch_norm_1 = tf.keras.layers.BatchNormalization(renorm=True)
        self.batch_norm_2 = tf.keras.layers.BatchNormalization(renorm=True)
        self.batch_norm_3 = tf.keras.layers.BatchNormalization(renorm=True)

        self.rab_conv_1_1 = tf.keras.layers.Conv2D(
            64, self.kernel_size, activation='swish', padding='same')
        self.rab_conv_1_2 = tf.keras.layers.Conv2D(
            64, self.kernel_size, activation='swish', padding='same')
        self.rab_conv_2_1 = tf.keras.layers.Conv2D(
            128, self.kernel_size, activation='swish', padding='same')
        self.rab_conv_2_2 = tf.keras.layers.Conv2D(
            128, self.kernel_size, activation='swish', padding='same')

        self.fa_conv_3_1 = tf.keras.layers.Conv2D(
            256, self.kernel_size, activation='swish', padding='same')
        self.fa_conv_3_2 = tf.keras.layers.Conv2D(
            256, self.kernel_size, activation='swish', padding='same')

        self.fa_conv_4_1 = tf.keras.layers.Conv2D(
            512, self.kernel_size, activation='swish', padding='same')
        self.fa_conv_4_2 = tf.keras.layers.Conv2D(
            512, self.kernel_size, activation='swish', padding='same')

        self.C_conv_1 = tf.keras.layers.Conv1D(
            1024, kernel_size=1, activation='swish', padding='valid')
        self.C_conv_2 = tf.keras.layers.Conv1D(
            1, kernel_size=1, activation='sigmoid', padding='valid')

    @tf.function
    def call(self, feat_tl, feat_tr, r_ab, is_testing=False):
        """ takes in r, t, output of encoder, r_ab and spit out correlation matrix features conf_1,2,3, class output G, and f_s1,2,3 """

        assert feat_tl.shape[1:] == (self.height // 2, self.width // 2, 1984)
        assert feat_tr.shape[1:] == (self.height // 2, self.width // 2, 1984)

        f_rab = self.rab_conv_1_2(self.rab_conv_1_1(r_ab))
        assert f_rab.shape[1:] == (self.height, self.width, 64)
        f_rab = self.batch_norm_1(self.rab_conv_2_2(
            self.rab_conv_2_1(f_rab[:, ::2, ::2, :])))
        assert f_rab.shape[1:] == (self.height // 2, self.width // 2, 128)

        C = self.correlate(feat_tl, feat_tr)
        conf = self.C_conv_1(C)
        assert conf.shape[1:] == (self.height*self.width // 4, 1024)
        conf = self.C_conv_2(conf)
        assert conf.shape[1:] == (self.height*self.width // 4, 1)

        f_a = self.attention(C, f_rab)
        f_a = tf.reshape(
            f_a, [f_a.shape[0], self.height // 2, self.width // 2, 128])
        assert f_a.shape[1:] == (self.height // 2, self.width // 2, 128)
        align_1 = f_a
        align_2 = self.batch_norm_2(self.fa_conv_3_2(
            self.fa_conv_3_1(align_1[:, ::2, ::2, :])))
        assert align_2.shape[1:] == (self.height // 4, self.width // 4, 256)
        align_3 = self.batch_norm_3(self.fa_conv_4_2(
            self.fa_conv_4_1(align_2[:, ::2, ::2, :])))
        assert align_3.shape[1:] == (self.height // 8, self.width // 8, 512)

        return (conf, align_1, align_2, align_3)

    @tf.function
    def correlate(self, t_lum, r_lum):
        t_flatten = tf.reshape(t_lum, [t_lum.shape[0], -1, t_lum.shape[-1]])
        r_flatten = tf.reshape(r_lum, [r_lum.shape[0], -1, r_lum.shape[-1]])
        t_flatten = t_flatten / \
            tf.norm(t_flatten, ord=2, axis=-1, keepdims=True)
        r_flatten = r_flatten / \
            tf.norm(r_flatten, ord=2, axis=-1, keepdims=True)

        corr = tf.keras.backend.batch_dot(r_flatten, t_flatten, axes=(-1, -1))
        assert corr.shape[1:] == (
            self.height * self.width // 4, self.height * self.width // 4)

        return corr

    @tf.function
    def attention(self, C, f_rab):
        corr = tf.nn.softmax(C/0.01, axis=1)
        rab_flatten = tf.reshape(f_rab, [f_rab.shape[0], -1, f_rab.shape[-1]])

        # [batch, sm(pixels1), pixels] * [batch, pixels1, filter] = [batch, pixels, filter]
        align = tf.keras.backend.batch_dot(corr, rab_flatten, axes=(1, 1))
        assert align.shape[1:] == (self.height * self.width // 4, 128)

        return align


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
            512, self.kernel_size, activation='swish', padding='same')
        self.conv_1_2 = tf.keras.layers.Conv2D(
            512, self.kernel_size, activation='swish', padding='same')
        self.conv_1_3 = tf.keras.layers.Conv2D(
            512, self.kernel_size, activation='swish', padding='same')

        self.conv_2_1 = tf.keras.layers.Conv2D(
            512, self.kernel_size, activation='swish', padding='same')
        self.conv_2_2 = tf.keras.layers.Conv2D(
            256, self.kernel_size, activation='swish', padding='same')
        self.conv_2_3 = tf.keras.layers.Conv2D(
            128, self.kernel_size, activation='swish', padding='same')

    @tf.function    
    def call(self, r_hist, is_testing=False):
        r_hist = tf.reshape(r_hist, [r_hist.shape[0], 1, 1, 441])
        conv_output = self.conv_1_1(r_hist)
        assert conv_output.shape[1:] == (1, 1, 512)
        conv_output = self.conv_1_2(conv_output)
        assert conv_output.shape[1:] == (1, 1, 512)
        conv_output = self.conv_1_3(conv_output)
        assert conv_output.shape[1:] == (1, 1, 512)

        f_1 = self.conv_2_1(conv_output)
        assert f_1.shape[1:] == (1, 1, 512)
        f_2 = self.conv_2_2(conv_output)
        assert f_2.shape[1:] == (1, 1, 256)
        f_3 = self.conv_2_3(conv_output)
        assert f_3.shape[1:] == (1, 1, 128)
        return f_1, f_2, f_3

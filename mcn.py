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
        self.conv_1 = tf.keras.layers.Conv2D(
            dim, 3, activation='relu', padding='same')
        self.batch_norm_1 = tf.keras.layers.BatchNormalization(renorm=True)
        self.conv_2 = tf.keras.layers.Conv2D(
            dim, 3, activation='relu', padding='same')
        self.batch_norm_2 = tf.keras.layers.BatchNormalization(renorm=True)

    def call(self, input):
        return input + self.batch_norm_2(self.conv_2(self.batch_norm_1(self.conv_1(input))))


""" 
    MCN Encoder 
    Input: luminance channel of input image and reference image
    Output: Feature matrix of input image t and reference image r
            un-concatenated feature convoluted matrix layer_2_6
"""


class Encoder(tf.keras.Model):
    def __init__(self, img_height, img_width):
        super(Encoder, self).__init__()

        self.height = img_height
        self.width = img_width

        """ Init layers """
        self.conv_1_1 = tf.keras.layers.Conv2D(
            64, 3, activation='relu', padding='same')
        self.conv_1_2 = tf.keras.layers.Conv2D(
            64, 3, activation='relu', padding='same')
        self.batch_norm_1 = tf.keras.layers.BatchNormalization(renorm=True)

        self.conv_2_1 = tf.keras.layers.Conv2D(
            128, 3, activation='relu', padding='same')
        self.conv_2_2 = tf.keras.layers.Conv2D(
            128, 3, activation='relu', padding='same')
        self.batch_norm_2 = tf.keras.layers.BatchNormalization(renorm=True)

        self.conv_3_1 = tf.keras.layers.Conv2D(
            256, 3, activation='relu', padding='same')
        self.conv_3_2 = tf.keras.layers.Conv2D(
            256, 3, activation='relu', padding='same')
        self.conv_3_3 = tf.keras.layers.Conv2D(
            256, 3, activation='relu', padding='same')
        self.batch_norm_3 = tf.keras.layers.BatchNormalization(renorm=True)

        self.conv_4_1 = tf.keras.layers.Conv2D(
            512, 3, activation='relu', padding='same')
        self.conv_4_2 = tf.keras.layers.Conv2D(
            512, 3, activation='relu', padding='same')
        self.conv_4_3 = tf.keras.layers.Conv2D(
            512, 3, activation='relu', padding='same')
        self.batch_norm_4 = tf.keras.layers.BatchNormalization(renorm=True)

        self.conv_5_1 = tf.keras.layers.Conv2D(
            512, 3, dilation_rate=2, activation='relu', padding='same')
        self.conv_5_2 = tf.keras.layers.Conv2D(
            512, 3, dilation_rate=2, activation='relu', padding='same')
        self.conv_5_3 = tf.keras.layers.Conv2D(
            512, 3, dilation_rate=2, activation='relu', padding='same')
        self.batch_norm_5 = tf.keras.layers.BatchNormalization(renorm=True)

        self.conv_6_1 = tf.keras.layers.Conv2D(
            512, 3, dilation_rate=2, activation='relu', padding='same')
        self.conv_6_2 = tf.keras.layers.Conv2D(
            512, 3, dilation_rate=2, activation='relu', padding='same')
        self.conv_6_3 = tf.keras.layers.Conv2D(
            512, 3, dilation_rate=2, activation='relu', padding='same')
        self.batch_norm_6 = tf.keras.layers.BatchNormalization(renorm=True)

        self.interpolate_3 = tf.keras.layers.UpSampling2D(
            size=2, interpolation='bilinear')
        self.interpolate_4 = tf.keras.layers.UpSampling2D(
            size=4, interpolation='bilinear')
        self.interpolate_5 = tf.keras.layers.UpSampling2D(
            size=4, interpolation='bilinear')
        self.interpolate_6 = tf.keras.layers.UpSampling2D(
            size=4, interpolation='bilinear')

    def encode(self, img_l):
        layer_1 = self.batch_norm_1(self.conv_1_2(self.conv_1_1(img_l)))
        assert layer_1.shape[1:] == (self.height, self.width, 64)

        layer_2 = self.batch_norm_2(self.conv_2_2(
            self.conv_2_1(layer_1[:, ::2, ::2, :])))
        assert layer_2.shape[1:] == (self.height // 2, self.width // 2, 128)

        layer_3 = self.batch_norm_3(self.conv_3_3(
            self.conv_3_2(self.conv_3_1(layer_2[:, ::2, ::2, :]))))
        assert layer_3.shape[1:] == (self.height // 4, self.width // 4, 256)

        layer_4 = self.batch_norm_4(self.conv_4_3(
            self.conv_4_2(self.conv_4_1(layer_3[:, ::2, ::2, :]))))
        assert layer_4.shape[1:] == (self.height // 8, self.width // 8, 512)

        layer_5 = self.batch_norm_5(self.conv_5_3(
            self.conv_5_2(self.conv_5_1(layer_4))))
        assert layer_5.shape[1:] == (self.height // 8, self.width // 8, 512)

        layer_6 = self.batch_norm_6(self.conv_6_3(
            self.conv_6_2(self.conv_6_1(layer_5))))
        assert layer_6.shape[1:] == (self.height // 8, self.width // 8, 512)

        inter_1 = layer_1[:, ::2, ::2, :]
        inter_2 = layer_2
        inter_3 = self.interpolate_3(layer_3)
        inter_4 = self.interpolate_4(layer_4)
        inter_5 = self.interpolate_5(layer_5)
        inter_6 = self.interpolate_6(layer_6)

        assert inter_1.shape[1:] == (self.height // 2, self.width // 2, 64)
        assert inter_2.shape[1:] == (self.height // 2, self.width // 2, 128)
        assert inter_3.shape[1:] == (self.height // 2, self.width // 2, 256)
        assert inter_4.shape[1:] == (self.height // 2, self.width // 2, 512)
        assert inter_5.shape[1:] == (self.height // 2, self.width // 2, 512)
        assert inter_6.shape[1:] == (self.height // 2, self.width // 2, 512)

        feat_l = tf.concat(
            [inter_1, inter_2, inter_3, inter_4, inter_5, inter_6], axis=-1)

        assert feat_l.shape[1:] == (self.height // 2, self.width // 2, 1984)

        return feat_l, layer_1, layer_2, layer_3, layer_6

    """ returns feature of r, feature of t, feature of t to calculate G_tl, layer_2_1, layer_2_2, layer_2_3 for skip connections """
    def call(self, r_l, t_l, is_testing=False):
        # encoding the reference image
        feat_rl, _, _, _, _ = self.encode(r_l)
        # encoding the target image
        feat_tl, layer_2_1, layer_2_2, layer_2_3, layer_2_6 = self.encode(t_l)

        return feat_rl, feat_tl, layer_2_6, layer_2_1, layer_2_2, layer_2_3


""" 
    MCN Decoder 1
    Input: output of color distribution module, 
           output of assignment module,
           features of corr matrix,
           output of encoder model,
           3rd layer of original convolution
    Output: first output,
            convoluted output of t

"""


class Decoder1(tf.keras.Model):
    def __init__(self):
        super(Decoder1, self).__init__()

        """ Init layers """
        self.resconv_1 = tf.keras.layers.Conv2D(
            512, 3, activation='relu', padding='same')
        self.batch_norm_1 = tf.keras.layers.BatchNormalization(renorm=True)
        self.deconv_up = tf.keras.layers.Conv2DTranspose(
            256, 4, strides=2, padding='same')
        self.deconv_short = tf.keras.layers.Conv2D(
            256, 3, padding='same', activation='relu')
        self.deconv_1 = tf.keras.layers.Conv2D(
            256, 3, padding='same', activation='relu')
        self.deconv_2 = tf.keras.layers.Conv2D(
            256, 3, padding='same', activation='relu')
        self.batch_norm_2 = tf.keras.layers.BatchNormalization(renorm=True)

        self.model_out = tf.keras.layers.Conv2D(
            2, 1, dilation_rate=1, activation='tanh')

        self.resblock_1 = ResBlock(512)
        self.resblock_2 = ResBlock(512)

    @tf.function
    def call(self, M1, enc_output_1, layer_3, is_testing=False):
        enc_output_1_global = enc_output_1 + M1
        resblock1 = self.batch_norm_1(self.resconv_1(enc_output_1_global))
        resblock2 = self.resblock_1(resblock1)
        resblock3 = self.resblock_2(resblock2)

        layer_up = self.deconv_up(resblock3) + self.deconv_short(layer_3)
        encoder_output = self.batch_norm_2(self.deconv_2(self.deconv_1(layer_up)))
        fake_img_1 = self.model_out(encoder_output)

        return encoder_output, fake_img_1


""" 
    MCN Decoder 2
    Input: output of color distribution module, 
           output of assignment module,
           features of corr matrix,
           output of encoder model,
           2rd layer of original convolution
    Output: first output,
            convoluted output of t
"""


class Decoder2(tf.keras.Model):
    def __init__(self):
        super(Decoder2, self).__init__()

        """ Init layers """
        self.resconv_1 = tf.keras.layers.Conv2D(
            256, 3, activation='relu', padding='same')
        self.batch_norm_1 = tf.keras.layers.BatchNormalization(renorm=True)
        self.deconv_up = tf.keras.layers.Conv2DTranspose(
            128, 4, strides=2, padding='same')
        self.deconv_short = tf.keras.layers.Conv2D(128, 3, padding='same')
        self.relu_1 = tf.keras.layers.ReLU()
        self.deconv = tf.keras.layers.Conv2D(128, 3, padding='same')
        self.relu_2 = tf.keras.layers.ReLU()
        self.batch_norm_2 = tf.keras.layers.BatchNormalization(renorm=True)

        self.model_out = tf.keras.layers.Conv2D(
            2, 1, dilation_rate=1, activation='tanh')

        self.resblock_1 = ResBlock(256)
        self.resblock_2 = ResBlock(256)
    
    @tf.function
    def call(self, M2, decoder_output1, layer_2, is_testing=False):
        enc_output_2_global = decoder_output1 + M2
        resblock1 = self.batch_norm_1(self.resconv_1(enc_output_2_global))
        resblock2 = self.resblock_1(resblock1)
        resblock3 = self.resblock_2(resblock2)
        
        layer_up = self.deconv_up(resblock3) + self.deconv_short(layer_2)
        decoder_output = self.batch_norm_2(self.relu_2(self.deconv(self.relu_1(layer_up))))
        fake_img_2 = self.model_out(decoder_output)

        return decoder_output, fake_img_2


""" 
    MCN Decoder 3
    Input: output of color distribution module, 
           output of assignment module,
           features of corr matrix,
           output of encoder model,
           1rd layer of original convolution
    Output: first output,
            convoluted output of t
"""


class Decoder3(tf.keras.Model):
    def __init__(self):
        super(Decoder3, self).__init__()

        """ Init layers """
        self.resconv_1 = tf.keras.layers.Conv2D(
            128, 3, activation='relu', padding='same')
        self.batch_norm_1 = tf.keras.layers.BatchNormalization(renorm=True)
        self.deconv_up = tf.keras.layers.Conv2DTranspose(
            128, 4, strides=2, padding='same')
        self.deconv_short = tf.keras.layers.Conv2D(128, 3, padding='same')
        self.relu = tf.keras.layers.ReLU()
        self.deconv = tf.keras.layers.Conv2D(128, 3, padding='same')
        self.leaky_relu = tf.keras.layers.LeakyReLU()

        self.model_out = tf.keras.layers.Conv2D(
            2, 1, dilation_rate=1, activation='tanh')

        self.resblock_2 = ResBlock(128)
        self.resblock_3 = ResBlock(128)

    @tf.function
    def call(self, M3, decoder_output2, layer_1, is_testing=False):
        conv9_3_global = decoder_output2 + M3
        conv9_resblock1 = self.batch_norm_1(self.resconv_1(conv9_3_global))
        conv9_resblock2 = self.resblock_2(conv9_resblock1)
        conv9_resblock3 = self.resblock_3(conv9_resblock2)
        
        layer_up = self.deconv_up(conv9_resblock3) + self.deconv_short(layer_1)
        decoder_output = self.leaky_relu(self.deconv(self.relu(layer_up)))
        fake_img_3 = self.model_out(decoder_output)

        return decoder_output, fake_img_3

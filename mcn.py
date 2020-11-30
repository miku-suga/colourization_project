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
    Output: Feature matrix of input image t and reference image r
            un-concatenated feature convoluted matrix layer_2_6
"""
class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()

        """ Init layers """
        self.padding_1 = tf.keras.layers.ZeroPadding2D(padding=(1,1))
        self.conv_1_1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')
        self.padding_2 = tf.keras.layers.ZeroPadding2D(padding=(1,1))
        self.conv_1_2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()

        self.padding_3 = tf.keras.layers.ZeroPadding2D(padding=(1,1))
        self.conv_2_1 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')
        self.padding_4 = tf.keras.layers.ZeroPadding2D(padding=(1,1))
        self.conv_2_2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()

        self.padding_5 = tf.keras.layers.ZeroPadding2D(padding=(1,1))
        self.conv_3_1 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')
        self.padding_6 = tf.keras.layers.ZeroPadding2D(padding=(1,1))
        self.conv_3_2 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')
        self.padding_7 = tf.keras.layers.ZeroPadding2D(padding=(1,1))
        self.conv_3_3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')
        self.batch_norm_3 = tf.keras.layers.BatchNormalization()

        self.padding_8 = tf.keras.layers.ZeroPadding2D(padding=(1,1))
        self.conv_4_1 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')
        self.padding_9 = tf.keras.layers.ZeroPadding2D(padding=(1,1))
        self.conv_4_2 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')
        self.padding_10 = tf.keras.layers.ZeroPadding2D(padding=(1,1))
        self.conv_4_3 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')
        self.batch_norm_4 = tf.keras.layers.BatchNormalization()

        self.padding_11 = tf.keras.layers.ZeroPadding2D(padding=(2,2))
        self.conv_5_1 = tf.keras.layers.Conv2D(512, 3, dilation=2, activation='relu', padding='same')
        self.padding_12 = tf.keras.layers.ZeroPadding2D(padding=(2,2))
        self.conv_5_2 = tf.keras.layers.Conv2D(512, 3, dilation=2, activation='relu', padding='same')
        self.padding_13 = tf.keras.layers.ZeroPadding2D(padding=(2,2))
        self.conv_5_3 = tf.keras.layers.Conv2D(512, 3, dilation=2, activation='relu', padding='same')
        self.batch_norm_5 = tf.keras.layers.BatchNormalization()

        self.padding_14 = tf.keras.layers.ZeroPadding2D(padding=(2,2))
        self.conv_6_1 = tf.keras.layers.Conv2D(512, 3, dilation=2, activation='relu', padding='same')
        self.padding_15 = tf.keras.layers.ZeroPadding2D(padding=(2,2))
        self.conv_6_2 = tf.keras.layers.Conv2D(512, 3, dilation=2, activation='relu', padding='same')
        self.padding_16 = tf.keras.layers.ZeroPadding2D(padding=(2,2))
        self.conv_6_3 = tf.keras.layers.Conv2D(512, 3, dilation=2, activation='relu', padding='same')
        self.batch_norm_6 = tf.keras.layers.BatchNormalization()    

        self.interpolate_1 = tf.keras.layers.UpSampling2D(size=0.5, interpolation='bilinear')
        self.interpolate_3 = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')
        self.interpolate_4 = tf.keras.layers.UpSampling2D(size=4, interpolation='bilinear')
        self.interpolate_5 = tf.keras.layers.UpSampling2D(size=4, interpolation='bilinear')
        self.interpolate_6 = tf.keras.layers.UpSampling2D(size=4, interpolation='bilinear')


    def call(self, r_l, t_l, m_1, m_2, m_3, is_testing=False):
        layer_1_1 = self.batch_norm_1(self.conv_1_2(self.padding_2(self.conv_1_1(self.padding_1(r_l)))))
        layer_1_2 = self.batch_norm_2(self.conv_2_2(self.padding_4(self.conv_2_1(self.padding_3(layer_1_1[:,:,::2,::2])))))
        layer_1_3 = self.batch_norm_3(self.conv_3_3(self.padding_7(self.conv_3_2(self.padding_6(self.conv_3_1(self.padding_5(layer_1_2[:,:,::2,::2])))))))
        layer_1_4 = self.batch_norm_4(self.conv_4_3(self.padding_10(self.conv_4_2(self.padding_9(self.conv_4_1(self.padding_8(layer_1_3[:,:,::2,::2])))))))
        layer_1_5 = self.batch_norm_5(self.conv_5_3(self.padding_13(self.conv_5_2(self.padding_12(self.conv_5_1(self.padding_11(layer_1_4)))))))
        layer_1_6 = self.batch_norm_6(self.conv_6_3(self.padding_16(self.conv_6_2(self.padding_15(self.conv_6_1(self.padding_14(layer_1_5)))))))

        t = tf.concat((self.interpolate_1(layer_1_1), layer_1_2, self.interpolate_3(layer_1_3), 
        self.interpolate_4(layer_1_4), self.interpolate_5(layer_1_5), self.interpolate_6(layer_1_6)), axis=1)

        layer_2_1 = self.batch_norm_1(self.conv_1_2(self.padding_2(self.conv_1_1(self.padding_1(t_l)))))
        layer_2_2 = self.batch_norm_2(self.conv_2_2(self.padding_4(self.conv_2_1(self.padding_3(layer_2_1[:,:,::2,::2])))))
        layer_2_3 = self.batch_norm_3(self.conv_3_3(self.padding_7(self.conv_3_2(self.padding_6(self.conv_3_1(self.padding_5(layer_2_2[:,:,::2,::2])))))))
        layer_2_4 = self.batch_norm_4(self.conv_4_3(self.padding_10(self.conv_4_2(self.padding_9(self.conv_4_1(self.padding_8(layer_2_3[:,:,::2,::2])))))))
        layer_2_5 = self.batch_norm_5(self.conv_5_3(self.padding_13(self.conv_5_2(self.padding_12(self.conv_5_1(self.padding_11(layer_2_4)))))))
        layer_2_6 = self.batch_norm_6(self.conv_6_3(self.padding_16(self.conv_6_2(self.padding_15(self.conv_6_1(self.padding_14(layer_2_5)))))))

        r = tf.concat((self.interpolate_1(layer_2_1), layer_2_2, self.interpolate_3(layer_2_3), 
        self.interpolate_4(layer_2_4), self.interpolate_5(layer_2_5), self.interpolate_6(layer_2_6)), axis=1)

        return r, t, layer_2_6
        """ ^^ feature of r, feature of t, feature of t to calculate G_tl, layer_2_1, layer_2_2, layer_2_3 for skip connections """

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
        self.padding_1 = tf.keras.layers.ZeroPadding2D(padding=(1,1))
        self.resconv_1 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.padding_2 = tf.keras.layers.ZeroPadding2D(padding=(1,1))
        self.deconv_up = tf.keras.layers.Conv2DTranspose(256, 4, strides=2, padding='same')
        self.padding_3 = tf.keras.layers.ZeroPadding2D(padding=(1,1))
        self.deconv_short = tf.keras.layers.Conv2D(256, 3, padding='same') 
        self.relu_1 = tf.keras.layers.ReLU()
        self.padding_4 = tf.keras.layers.ZeroPadding2D(padding=(1,1))
        self.deconv_1 = tf.keras.layers.Conv2D(256, 3, padding='same')
        self.relu_2 = tf.keras.layers.ReLU()
        self.padding_5 = tf.keras.layers.ZeroPadding2D(padding=(1,1))
        self.deconv_2 = tf.keras.layers.Conv2D(256, 3, padding='same')
        self.relu_3 = tf.keras.layers.ReLU()
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()

        self.model_out = tf.keras.layers.Conv2D(2, 1, dilation=1, activation='tanh')

        self.resblock_1 = ResBlock(512)
        self.resblock_2 = ResBlock(512)

    def call(self, distri, assign, corr_feat_3, enc_output_1, layer_3, is_testing=False):
        conv_global1_repeat = distri.broadcast_to(enc_output_1)
        enc_output_1_global = enc_output_1 + assign * corr_feat_3 + conv_global1_repeat * (1 - corr_feat_3)
        layer_1 = self.resblock_2(self.resblock_1(self.batch_norm_1(self.resconv_1(self.padding_1(enc_output_1_global)))))
        layer_up = self.deconv_up(self.padding_2(layer_1)) + self.deconv_short(self.padding_3(layer_3))
        encoder_output = self.batch_norm_1(self.relu_3(self.deconv_2(self.padding_5(self.relu_2(self.deconv_1(self.padding_4(self.relu_1(layer_up))))))))
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
        self.padding_1 = tf.keras.layers.ZeroPadding2D(padding=(1,1))
        self.resconv_1 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.padding_2 = tf.keras.layers.ZeroPadding2D(padding=(1,1))
        self.deconv_up = tf.keras.layers.Conv2DTranspose(128, 4, strides=2, padding='same')
        self.padding_3 = tf.keras.layers.ZeroPadding2D(padding=(1,1))
        self.deconv_short = tf.keras.layers.Conv2D(128, 3, padding='same') 
        self.relu_1 = tf.keras.layers.ReLU()
        self.padding_4 = tf.keras.layers.ZeroPadding2D(padding=(1,1))
        self.deconv = tf.keras.layers.Conv2D(128, 3, padding='same') 
        self.relu_2 = tf.keras.layers.ReLU()
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()

        self.model_out = tf.keras.layers.Conv2D(2, 1, dilation=1, activation='tanh')

        self.resblock_1 = ResBlock(256)
        self.resblock_2 = ResBlock(256)

    def call(self, distri, assign, corr_feat_2, enc_output_2, layer_2, is_testing=False):
        conv_global2_repeat = distri.broadcast_to(enc_output_2)
        enc_output_2_global = enc_output_2 + assign * corr_feat_2 + conv_global2_repeat * (1 - corr_feat_2)
        layer_1 = self.resblock_2(self.resblock_1(self.batch_norm_1(self.resconv_1(self.padding_1(enc_output_2_global)))))
        layer_up = self.deconv_up(self.padding_2(layer_1)) + self.deconv_short(self.padding_3(layer_2))
        encoder_output = self.batch_norm_1(self.relu_2(self.deconv_1(self.padding_4(self.relu_1(layer_up)))))
        fake_img_2 = self.model_out(encoder_output)
        
        return encoder_output, fake_img_2

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
        self.padding_1 = tf.keras.layers.ZeroPadding2D(padding=(1,1))
        self.resconv_1 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.padding_2 = tf.keras.layers.ZeroPadding2D(padding=(1,1))
        self.deconv_up = tf.keras.layers.Conv2DTranspose(128, 4, strides=2, padding='same')
        self.padding_3 = tf.keras.layers.ZeroPadding2D(padding=(1,1))
        self.deconv_short = tf.keras.layers.Conv2D(128, 3, padding='same') 
        self.relu = tf.keras.layers.ReLU()
        self.padding_4 = tf.keras.layers.ZeroPadding2D(padding=(1,1))
        self.deconv = tf.keras.layers.Conv2D(128, 3, padding='same') 
        self.leaky_relu = tf.keras.layers.LeakyReLU()

        self.model_out = tf.keras.layers.Conv2D(2, 1, dilation=1, activation='tanh')

        self.resblock_1 = ResBlock(128)
        self.resblock_2 = ResBlock(128)

    def call(self, distri, assign, corr_feat_1, enc_output_1, layer_1, is_testing=False):
        conv_global3_repeat = distri.broadcast_to(enc_output_1)
        enc_output_3_global = enc_output_1 + assign * corr_feat_1 + conv_global3_repeat * (1 - corr_feat_1)
        layer_1 = self.resblock_2(self.resblock_1(self.batch_norm_1(self.resconv_1(self.padding_1(enc_output_3_global)))))
        layer_up = self.deconv_up(self.padding_2(layer_1)) + self.deconv_short(self.padding_3(layer_1))
        encoder_output = self.leaky_relu(self.deconv_(self.padding_4(self.relu(layer_up))))
        fake_img_3 = self.model_out(encoder_output)
        
        return encoder_output, fake_img_3


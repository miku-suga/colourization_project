import tensorflow as tf
import numpy as np
from gcfn import GatedFusionModule, SemanticAssignmentModule, ColorDistributionModule
from mcn import Encoder, Decoder1, Decoder2, Decoder3


class Model(tf.keras.Model):
    def __init__(self, num_classes, img_height, img_width):
        super(Model, self).__init__()

        self.height = img_height
        self.width = img_width
        self.num_classes = num_classes
        
        self.encoder = Encoder()
        self.decoder_1 = Decoder1()
        self.decoder_2 = Decoder2()
        self.decoder_3 = Decoder3()
        self.cdm = ColorDistributionModule()
        self.sam = SemanticAssignmentModule(num_classes, img_height, img_width)
        self.gfm = GatedFusionModule(img_height, img_width)

        self.batch_size_1 = 48
        self.batch_size_2 = 12

        self.pixel_weight = 1000
        self.hist_weight = 1
        self.class_weight = 1
        self.g_weight = 0.1
        self.tv_weight = 10

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
        self.discrim = Discriminator()

    def call(self, r_hist, r_ab, r_l, t_l, is_testing=False):
        """ get features and output of convolution layers from encoder """
        f_rl, f_tl, enc_output_1, layer_1, layer_2, layer_3 = self.encoder(
            r_l, t_l)

        f_global1, f_global2, f_global3 = self.cdm(r_hist)
        g_tl, corr, f_s1, f_s2, f_s3 = self.sam(f_tl, f_rl, r_ab)

        gate_out1, gate_out2, gate_out3 = self.gfm(
            corr, f_s1, f_s2, f_s3, f_global1, f_global2, f_global3)

        decoder_output1, fake_img_1 = self.decoder_1(
            gate_out1, enc_output_1, layer_3)
        decoder_output2, fake_img_2 = self.decoder_2(
            gate_out2, decoder_output1, layer_2)
        decoder_output3, fake_img_3 = self.decoder_3(
            gate_out3, decoder_output2, layer_1)

        return g_tl, fake_img_1, fake_img_2, fake_img_3


    """ 
        Classification Loss Function 
        Input: g_tl and G label g_tl_label
        Output: Loss
    """

    def loss_class(self, g_tl, g_tl_label):
        loss_func = tf.keras.losses.CategoricalCrossentropy()
        return loss_func(g_tl_label, g_tl).numpy()

    """ 
        Smooth L1 Loss Function 
        Input: Output image t_ab and target label t_label
        Output: Loss
    """

    def loss_pixel(self, t_ab, t_label):
        return tf.losses.huber_loss(t_label, t_ab)

    """ 
        Histogram Loss Function 
        Input: Histogram of r and histogram of t
        Output: Loss
    """

    def loss_hist(self, r_h, t_h):
        return 2 * tf.reduce_sum(tf.divide(tf.square(t_h - r_h), (t_h + r_h)))

    """ 
        TV REGULARIZATION Loss Function 
        Input: t_ab
        Output: variation
    """

    def loss_tv(self, t_ab):
        return tf.image.total_variation(t_ab)

    def loss_G(self, t_lab):
        result = 0.5 * E ((self.discrim(t_lab) - 1) ** 2)   # have to figure out what's E
        return tf.math.reduce_min(result)


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.kernel_size = 3

        self.conv_1_1 = tf.keras.layers.Conv2D(
            32, self.kernel_size, activation='relu', padding='same')
        self.conv_1_2 = tf.keras.layers.Conv2D(
            32, self.kernel_size, activation='relu', padding='same')
        self.conv_2_1 = tf.keras.layers.Conv2D(
            64, self.kernel_size, activation='relu', padding='same')
        self.conv_2_2 = tf.keras.layers.Conv2D(
            64, self.kernel_size, activation='relu', padding='same')
        self.conv_3_1 = tf.keras.layers.Conv2D(
            128, self.kernel_size, activation='relu', padding='same')
        self.conv_3_2 = tf.keras.layers.Conv2D(
            128, self.kernel_size, activation='relu', padding='same')
        self.conv_4_1 = tf.keras.layers.Conv2D(
            256, self.kernel_size, activation='relu', padding='same')
        self.conv_4_2 = tf.keras.layers.Conv2D(
            256, self.kernel_size, activation='relu', padding='same')

        self.dense_1 = tf.keras.layers.Dense(512, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(512, activation='relu')
        self.dense_3 = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, images):
        out = self.conv_1_1(images)
        out = self.conv_1_2(out)
        out = tf.nn.max_pool2d(out, 2, strides=2, padding='VALID')
        out = self.conv_2_1(out)
        out = self.conv_2_2(out)
        out = tf.nn.max_pool2d(out, 2, strides=2, padding='VALID')
        out = self.conv_3_1(out)
        out = self.conv_3_2(out)
        out = tf.nn.max_pool2d(out, 2, strides=2, padding='VALID')
        out = self.conv_4_1(out)
        out = self.conv_4_2(out)
        out = tf.nn.max_pool2d(out, 2, strides=2, padding='VALID')

        out = tf.reshape(out, [out.shape[0], -1])
        out = self.dense_1(out)
        out = self.dense_2(out)
        out = self.dense_3(out)

        return out

    def loss(self):
        pass

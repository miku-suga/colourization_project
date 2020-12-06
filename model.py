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

        self.encoder = Encoder(img_height, img_width)
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

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def call(self, r_hist, r_ab, r_l, t_l, is_testing=False):
        """ get features and output of convolution layers from encoder """
        feat_rl, feat_tl, enc_output, layer_1, layer_2, layer_3 = self.encoder(
            r_l, t_l)

        f_global1, f_global2, f_global3 = self.cdm(r_hist)
        g_tl, corr, align_1, align_2, align_3 = self.sam(
            feat_tl, feat_rl, r_ab, enc_output)

        gate_out1, gate_out2, gate_out3 = self.gfm(
            corr, align_1, align_2, align_3, f_global1, f_global2, f_global3)

        decoder_output1, fake_img_1 = self.decoder_1(
            gate_out1, enc_output, layer_3)
        decoder_output2, fake_img_2 = self.decoder_2(
            gate_out2, decoder_output1, layer_2)
        _, fake_img_3 = self.decoder_3(
            gate_out3, decoder_output2, layer_1)

        return g_tl, fake_img_1, fake_img_2, fake_img_3

    def loss_function(self, t_ab_real, t_ab_out_1, t_ab_out_2, t_ab_out_3, r_h, t_h_out_1, t_h_out_2, t_h_out_3, discrim_logits, g_tl_out, g_tl_real, is_first_round):
        # Classification Loss Function
        loss_class = self.class_weight * tf.nn.sparse_softmax_cross_entropy_with_logits(g_tl_real, g_tl_out)

        # Smooth L1 Loss Function
        """ loss_pixel_1 = self.pixel_weight * (tf.keras.losses.Huber())(t_ab_real, t_ab_out_1)
        loss_pixel_2 = self.pixel_weight * (tf.keras.losses.Huber())(t_ab_real, t_ab_out_2)
        loss_pixel_3 = self.pixel_weight * (tf.keras.losses.Huber())(t_ab_real, t_ab_out_3) """
        loss_pixel = self.pixel_weight * (tf.keras.losses.Huber())(t_ab_real, t_ab_out_3)

        # Histogram Loss Function
        """ loss_hist_1 = self.hist_weight * 2 * \
            tf.reduce_sum(tf.divide(tf.square(t_h_out_1 - r_h), (t_h_out_1 + r_h + 0.1)))
        loss_hist_2 = self.hist_weight * 2 * \
            tf.reduce_sum(tf.divide(tf.square(t_h_out_2 - r_h), (t_h_out_2 + r_h + 0.1)))
        loss_hist_3 = self.hist_weight * 2 * \
            tf.reduce_sum(tf.divide(tf.square(t_h_out_3 - r_h), (t_h_out_3 + r_h + 0.1))) """
        loss_hist = self.hist_weight * 2 * \
            tf.reduce_sum(tf.divide(tf.square(t_h_out_3 - r_h), (t_h_out_3 + r_h + 0.1)))

        # TV REGULARIZATION Loss Function
        loss_tv_1 = self.tv_weight * tf.image.total_variation(t_ab_out_1)
        loss_tv_2 = self.tv_weight * tf.image.total_variation(t_ab_out_2)
        loss_tv_3 = self.tv_weight * tf.image.total_variation(t_ab_out_3)

        # generator loss
        loss_g = self.g_weight * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(discrim_logits), logits=discrim_logits))

        """ print('loss_class', loss_class)
        print('loss_pixel', loss_pixel.numpy())
        print('loss_hist', loss_hist)
        print('loss_tv', loss_tv)
        print('loss_g', loss_g) """

        if is_first_round:
            loss = ((loss_pixel + loss_tv_1) / self.batch_size_1) + loss_g 
            loss += ((loss_pixel + loss_tv_2) / self.batch_size_1) + loss_g 
            loss += ((loss_pixel + loss_tv_3) / self.batch_size_1) + loss_g 
            loss += loss_class
        else:
            loss = ((loss_pixel + loss_tv_1 + loss_hist) / self.batch_size_2) + loss_g 
            loss += ((loss_pixel + loss_tv_2 + loss_hist) / self.batch_size_2) + loss_g 
            loss += ((loss_pixel + loss_tv_3 + loss_hist) / self.batch_size_2) + loss_g 
            loss += loss_class

        print('loss', loss)

        return loss


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

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

        self.dense_1 = tf.keras.layers.Dense(512)
        self.dense_1_relu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.dense_2 = tf.keras.layers.Dense(512)
        self.dense_2_relu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.dense_3 = tf.keras.layers.Dense(1)

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
        out = self.dense_1_relu(self.dense_1(out))
        out = self.dense_2_relu(self.dense_2(out))
        out = self.dense_3(out)

        return out

    @tf.function
    def loss_function(self, logits_fake, logits_real):
        D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(logits_fake), logits=logits_fake))
        D_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(logits_real), logits=logits_real))
        return D_loss

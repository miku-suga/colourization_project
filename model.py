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
        self.sam = SemanticAssignmentModule(img_height, img_width)
        self.gfm = GatedFusionModule(img_height, img_width)
        self.classify = ClassifyImg(num_classes)

        self.batch_size_1 = tf.constant(30.0)
        self.batch_size_2 = tf.constant(8.0)

        self.pixel_weight = tf.constant(1000.0)
        self.hist_weight = tf.constant(1.0)
        self.class_weight = tf.constant(1.0)
        self.g_weight = tf.constant(5.0)
        self.tv_weight = tf.constant(0.0)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

    @tf.function
    def call(self, r_hist, r_ab, r_l, t_l, noRef=False):
        """ get features and output of convolution layers from encoder """
        if noRef:
            feat_rl, feat_tl, enc_output, layer_1, layer_2, layer_3 = self.encoder(
                r_l, t_l)
            g_tl = self.classify(enc_output)
            decoder_output1, fake_img_1 = self.decoder_1(
                0.0, enc_output, layer_3)
            decoder_output2, fake_img_2 = self.decoder_2(
                0.0, decoder_output1, layer_2)
            _, fake_img_3 = self.decoder_3(0.0, decoder_output2, layer_1)
        else:
            feat_rl, feat_tl, enc_output, layer_1, layer_2, layer_3 = self.encoder(
                r_l, t_l)
            g_tl = self.classify(enc_output)
            f_global1, f_global2, f_global3 = self.cdm(r_hist)
            conf, align_1, align_2, align_3 = self.sam(
                feat_tl, feat_rl, r_ab)

            gate_out1, gate_out2, gate_out3 = self.gfm(
                conf, align_1, align_2, align_3, f_global1, f_global2, f_global3)

            decoder_output1, fake_img_1 = self.decoder_1(
                gate_out1, enc_output, layer_3)
            decoder_output2, fake_img_2 = self.decoder_2(
                gate_out2, decoder_output1, layer_2)
            _, fake_img_3 = self.decoder_3(gate_out3, decoder_output2, layer_1)

        return g_tl, fake_img_1, fake_img_2, fake_img_3

    @tf.function
    def loss_function(self, t_ab_real, t_ab_out_1, t_ab_out_2, t_ab_out_3, r_h, t_h_out_1, t_h_out_2, t_h_out_3, discrim_logits, g_tl_out, g_tl_real, is_first_round):
        # Classification Loss Function
        loss_class = self.class_weight * \
            tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(g_tl_real, g_tl_out))

        # Smooth L1 Loss Function
        loss_pixel_1 = self.pixel_weight * \
            (tf.keras.losses.Huber())(tf.nn.max_pool(
                t_ab_real, 4, 4, 'SAME'), t_ab_out_1)
        loss_pixel_2 = self.pixel_weight * \
            (tf.keras.losses.Huber())(tf.nn.max_pool(
                t_ab_real, 2, 2, 'SAME'), t_ab_out_2)
        loss_pixel_3 = self.pixel_weight * \
            (tf.keras.losses.Huber())(t_ab_real, t_ab_out_3)

        # Histogram Loss Function
        loss_hist_3 = 0.0
        if not is_first_round:
            loss_hist_1 = self.hist_weight * 2 * \
                tf.reduce_sum(tf.math.divide_no_nan(
                    tf.square(t_h_out_1 - r_h), (t_h_out_1 + r_h)))
            loss_hist_2 = self.hist_weight * 2 * \
                tf.reduce_sum(tf.math.divide_no_nan(
                    tf.square(t_h_out_2 - r_h), (t_h_out_2 + r_h)))
            loss_hist_3 = self.hist_weight * 2 * \
                tf.reduce_sum(tf.math.divide_no_nan(
                    tf.square(t_h_out_3 - r_h), (t_h_out_3 + r_h)))

        # TV REGULARIZATION Loss Function
        loss_tv_1 = self.tv_weight * \
            tf.reduce_mean(tf.image.total_variation(t_ab_out_1))
        loss_tv_2 = self.tv_weight * \
            tf.reduce_mean(tf.image.total_variation(t_ab_out_2))
        loss_tv_3 = self.tv_weight * \
            tf.reduce_mean(tf.image.total_variation(t_ab_out_3))

        # generator loss
        loss_g = - self.g_weight * tf.reduce_mean(discrim_logits)

        tf.print('wighted loss_class', loss_class)
        tf.print('wighted loss_pixel', loss_pixel_1, loss_pixel_2, loss_pixel_3)
        tf.print('wighted loss_hist', loss_hist_3)
        tf.print('wighted loss_tv', loss_tv_3)
        tf.print('wighted loss_g', loss_g)

        if is_first_round:
            loss = loss_pixel_1 + loss_tv_1
            loss += loss_pixel_2 + loss_tv_2
            loss += loss_pixel_3 + loss_tv_3
            loss += loss_class + loss_g
        else:
            loss = loss_pixel_1 + loss_tv_1 + loss_hist_1/self.batch_size_2
            loss += loss_pixel_2 + loss_tv_2 + loss_hist_2/self.batch_size_2
            loss += loss_pixel_3 + loss_tv_3 + loss_hist_3/self.batch_size_2
            loss += loss_class + loss_g

        return loss, loss_class

class ClassifyImg(tf.keras.Model):
    def __init__(self, num_classes):
        super(ClassifyImg, self).__init__()

        self.gtl_dense_1 = tf.keras.layers.Dense(512, activation='swish')
        self.gtl_dense_2 = tf.keras.layers.Dense(num_classes)

    @tf.function
    def call(self, gtl_input):
        mp = tf.nn.max_pool2d(gtl_input, (gtl_input.shape[1], gtl_input.shape[2]),
                              strides=1, padding="VALID")
        assert mp.shape[1:] == (1, 1, 512)

        g_tl = self.gtl_dense_1(tf.reshape(mp, [mp.shape[0], -1]))
        assert g_tl.shape[1:] == (512)
        g_tl = self.gtl_dense_2(g_tl)

        return g_tl

class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0004)

        self.kernel_size = 3

        self.conv_1_1 = tf.keras.layers.Conv2D(
            32, self.kernel_size, activation='swish', padding='same')
        self.conv_1_2 = tf.keras.layers.Conv2D(
            32, self.kernel_size, activation='swish', padding='same')
        self.conv_2_1 = tf.keras.layers.Conv2D(
            64, self.kernel_size, activation='swish', padding='same')
        self.conv_2_2 = tf.keras.layers.Conv2D(
            64, self.kernel_size, activation='swish', padding='same')
        self.conv_3_1 = tf.keras.layers.Conv2D(
            128, self.kernel_size, activation='swish', padding='same')
        self.conv_3_2 = tf.keras.layers.Conv2D(
            128, self.kernel_size, activation='swish', padding='same')
        self.conv_4_1 = tf.keras.layers.Conv2D(
            256, self.kernel_size, activation='swish', padding='same')
        self.conv_4_2 = tf.keras.layers.Conv2D(
            256, self.kernel_size, activation='swish', padding='same')

        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.dense_1 = tf.keras.layers.Dense(512)
        self.dense_1_relu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.dense_2 = tf.keras.layers.Dense(512)
        self.dense_2_relu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.dense_3 = tf.keras.layers.Dense(1)

    @tf.function
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
        out = self.layer_norm(out)
        out = self.dense_1_relu(self.dense_1(out))
        out = self.dense_2_relu(self.dense_2(out))
        out = self.dense_3(out)

        return out

    def gradient_penalty(self, data_fake, data_real):
        alpha = tf.random.uniform(
            shape=[len(data_real), 1, 1, 1], 
            minval=0.,
            maxval=1.
        )

        interpolates = data_real + alpha * (data_fake - data_real)
        with tf.GradientTape() as t:
            t.watch(interpolates)
            pred = self(interpolates)
            
        gradients = t.gradient(pred, [interpolates])[0]

        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)

        return gradient_penalty

    # From WGAN-GP (Improved Training of Wasserstein GANs) paper
    @tf.function
    def loss_function(self, logits_fake, logits_real, gradient_penalty):
        LAMBDA = 10

        disc_cost = tf.reduce_mean(logits_fake) - tf.reduce_mean(logits_real)
        disc_cost += LAMBDA*gradient_penalty

        return disc_cost

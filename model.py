import tensorflow as tf
import numpy as np
from gcfn import GatedFusionModule, SemanticAssignmentModule, ColorDistributionModule
from mcn import Encoder, Decoder1, Decoder2, Decoder3


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

        self.encoder = Encoder()
        self.decoder_1 = Decoder1()
        self.decoder_2 = Decoder2()
        self.decoder_3 = Decoder3()
        self.cdm = ColorDistributionModule()
        self.sam = SemanticAssignmentModule()
        self.gfm = GatedFusionModule()

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

    def loss_G(self, t_ab, t_l):
        pass


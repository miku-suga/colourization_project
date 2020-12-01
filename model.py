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
        f_rl, f_tl, conv_out = self.encoder(r_l, t_l)
        
        f_global1, f_global2, f_global3 = self.cdm(r_hist)
        g_tl, corr, f_s1, f_s2, f_s3 = self.sam(f_twannal, f_rl, r_ab)
        
        m1, m2, m3 = self.gfm(corr, f_s1, f_s2, f_s3, f_global1, f_global2, f_global3)
        
        encoder_output, fake_img_1 = self.Decoder1(distri, assign, corr_feat_3, enc_output_1, layer_3)
        encoder_output, fake_img_2 = self.Decoder2(distri, assign, corr_feat_2, enc_output_2, layer_2)
        encoder_output, fake_img_3 = self.Decoder3(distri, assign, corr_feat_1, enc_output_1, layer_1)

        return g_tl, fake_img_1, fake_img_2, fake_img_3

    """ 
        Loss Function 
        Input: Input class label, G
               3 ab channel matrices of input image T
               luminance channel of input image T
        Output: Loss
    """ 
    def loss(self, g, t_ab_1, t_ab_2, t_ab_3, t_l):

        def loss_class(g_tl):
            pass

        def loss_pixel(t_ab):
            pass

        def loss_hist(t_ab):
            pass

        def loss_tv(t_ab):
            pass

        def loss_G(t_ab, t_l):
            pass
        
        pass
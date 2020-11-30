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

    def call(self, r_hist, r_ab, r_l, t_l, is_testing=False):
        """ get features and output of convolution layers from encoder """
        r, t, conv_output = self.encoder(r_l, t_l)

        """ Use r and t to get correlation matrix, and do conf feature to get conf_1,2,3 """

        """ get align_1,2,3 from assignment module """

        """ get conv_global1,2,3 from color distribution module """
        conv_global1, conv_global2, conv_global3 = self.cdm(r_hist)

        """ get class_output from assignment module  """


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
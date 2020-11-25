import tensorflow as tf
import numpy as np
from gcfn import GatedFusionModule, SemanticAssignmentModule, ColorDistributionModule
from mcn import Encoder, Decoder1, Decoder2, Decoder3
from tensorflow.keras import Model

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        pass

    def call(self, inputs, is_testing=False):
        pass

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
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from preprocess import get_tf_dataset, create_dict
from model import Model
from matplotlib import pyplot as plt
from util import calc_hist


def calc_total_loss(classification_loss, smoothL1_los, hist_loss, tv_reg_loss, generator_loss):
    #TODO:
    pass

def trainMCN(model, ref_dict, target_dict, target_l_dict):
    r_hist, r_ab, r_l = ref_dict["r_hist"], ref_dict["r_ab"], ref_dict['r_l']
    t_hist, t_ab, t_l = target_dict["t_hist"], target_dict["t_ab"], target_dict["t_l"]
    t_lhist, t_lab, t_ll = target_l_dict["r_hist"], target_l_dict["r_ab"], target_l_dict["r_l"]

    num_inputs = len(train_data)
    for i in range(0, num_inputs - model.batch_size_1, model.batch_size)_1:
        batch_r_hist = r_hist[i:i+model.batch_size_1]
        batch_r_ab = r_ab[i:i+model.batch_size_1]
        batch_r_l = r_l[i:i+model.batch_size_1]

        batch_t_hist = t_hist[i:i+model.batch_size_1]
        batch_t_ab = t_ab[i:i+model.batch_size_1]
        batch_t_l = t_l[i:i+model.batch_size_1]
        
        batch_t_lhist = t_label_ab[i:i+model.batch_size_1]
        batch_t_lab = t_label_ab[i:i+model.batch_size_1]
        batch_t_ll = t_label_l[i:i+model.batch_size_1]

        with tf.GradientTape() as tape:
            g_tl, fake_img_1, fake_img_2, fake_img_3 = model(batch_r_hist, batch_r_ab, batch_r_l, batch_t_l)
            classification_loss = model.loss_class(g_tl, batch_t_ll)
            smoothL1_loss = model.loss_pixel(batch_t_ab, batch_t_lab)
            hist_loss = 0
            tv_reg_loss = loss_tv(batch_t_ab)
            generator_loss = loss_G(batch_t_lab)

            total_loss = calc_total_loss(classification_loss, smoothL1_los, hist_loss, tv_reg_loss, generator_loss)
        
        gradients = tape.gradient(total_loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def train_everything(model, reference_dict, target_dict, target_label_dict):
    #TODO:
    pass


def visualize_results(images):
    fig = plt.figure()
    for i in range(len(images)):
            ax = fig.add_subplot(i, 1, 1)
            ax.imshow(images[i], cmap="Greys")
    plt.show()

def main():
    """ num_threads = 0
    serial_batches = True
    no_flip = True
    display_id = -1 """

    image_height = 64
    image_width = 64
    num_classes = 10

    train_data, test_data = get_tf_dataset()

    train_ref = train_data[:]
    train_target = get_target(train_data)
    

    # call function to turn train_data/label into a dict consisting of l, ab, hist
    ref_dict, target_dict = create_dict(train_ref, train_target)

    # create model
    model = Model(num_classes, image_height, image_width)


    # We are going to use target label as the train reference.
    # train MCN without histogram loss
    trainMCN(model, ref_dict, target_dict, ref_dict)



    # train everything 
    train_everything(model, ref_dict_list, target_dict_list, target_label_dict) 

    # call model on first 10 test examples
    test_target = target_dict_list[:10]
    test_ref = ref_dict_list[:10]
    g_tl, fake_img_1, fake_img_2, fake_img_3 = model(test_ref['hist'], test_ref['ab'], test_ref['l'], test_target['l'])

    # visualize
    visualize_results(fake_img_3)


if __name__ == '__main__':
    main()
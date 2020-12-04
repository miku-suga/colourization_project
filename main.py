import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from preprocess import get_tf_dataset, create_dict
from model import Model
from matplotlib import pyplot as plt
from util import calc_hist

# will adjust the parameters accordingly - miku
def trainMCN(model, reference_dict, target_dict, target_label_dict):
    r_hist, r_ab, r_l = reference_dict["hist"], reference_dict["ab"], reference_dict["l"]
    t_hist, t_ab, t_l = target_dict["hist"], target_dict["ab"], target_dict["l"]

    num_inputs = len(train_data)
    for i in range(0, num_inputs - model.batch_size_1, model.batch_size):
        batch_r_hist = r_hist[i:i+model.batch_size_1]
        batch_r_ab = r_ab[i:i+model.batch_size_1]
        batch_r_l = r_l[i:i+model.batch_size_1]

        batch_t_hist = t_hist[i:i+model.batch_size_1]
        batch_t_ab = t_ab[i:i+model.batch_size_1]
        batch_t_l = t_l[i:i+model.batch_size_1]
        
        batch_t_label_l = t_label_l[i:i+model.batch_size_1]
        batch_t_label_ab = t_label_ab[i:i+model.batch_size_1]

        with tf.GradientTape() as tape:
            g_tl, fake_img_1, fake_img_2, fake_img_3 = model(batch_r_hist, batch_r_ab, batch_r_l, batch_t_l)
            classification_loss = model.loss_class(g_tl, batch_t_label_l)
            smoothL1_loss = model.loss_pixel(batch_t_ab, batch_t_label_ab)
            hist_loss = model.loss_hist(batch_r_hist, batch_t_hist)
            tv_reg_loss = loss_tv(batch_t_ab)

            #IM STILL FIGURING OUT how to get the D value -miku
            pass
        
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def train_everything(model, reference_dict, target_dict, target_label_dict):
    #working on it - miku
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

    target_list, ref_list = get_tf_dataset()

    # call function to turn train_data/label into a dict consisting of l, ab, hist
    target_dict_list, ref_dict_list = create_dict(target_list, ref_list)

    # create model
    model = Model(num_classes, image_height, image_width)

    # train MCN without histogram loss
    target_label_dict = None
    trainMCN(model, ref_dict_list, target_dict_list, target_label_dict)

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
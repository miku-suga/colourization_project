import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from preprocess import get_tf_dataset, IMAGE_SIZE
from model import Model
from matplotlib import pyplot as plt
from util import calc_hist


def calc_total_loss(model, classification_loss, smoothL1_los, hist_loss, tv_reg_loss, generator_loss):
    total = 0
    total += model.pixel_weight * smoothL1_los
    total += model.hist_weight * hist_loss
    total += model.class_weight * classification_loss
    total += model.g_weight * generator_loss
    total += model.tv_weight * tv_reg_loss
    return total

def trainMCN(model, ref_data, target_data, noRef=False):
    for ref_batch, target_batch in zip(ref_data.as_numpy_iterator(), target_data.as_numpy_iterator()):
        r_l, r_ab, r_hist, r_label = ref_batch
        t_l, t_ab, t_hist, t_label = target_batch

        with tf.GradientTape() as tape:
            # TODO: review this
            g_tl, fake_img_1, fake_img_2, fake_img_3 = model(r_hist, r_ab, r_l, t_l)
            classification_loss = model.loss_class(g_tl, t_label)
            smoothL1_loss = model.loss_pixel(r_ab, t_ab)
            hist_loss = 0
            tv_reg_loss = model.loss_tv(t_ab)
            generator_loss = model.loss_G(t_ab)

            total_loss = calc_total_loss(classification_loss, smoothL1_loss, hist_loss, tv_reg_loss, generator_loss)
        
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
    num_classes = 365
    batch_size = 32
    training_size = 100
    testing_size = 10

    train_target_data = get_tf_dataset(batch_size, 'train', training_size)
    train_ref_data = get_tf_dataset(batch_size, 'train', training_size) # TODO: fix this?
    test_data = get_tf_dataset(batch_size, 'test', testing_size)

    model = Model(num_classes, IMAGE_SIZE, IMAGE_SIZE)

    # We are going to use target label as the train reference.
    # train MCN without histogram loss
    trainMCN(model, train_target_data, train_target_data, noRef=True)

    # train everything 
    trainMCN(model, train_ref_data, train_target_data)

    # # call model on first 10 test examples
    # test_target = target_dict_list[:10]
    # test_ref = ref_dict_list[:10]
    # g_tl, fake_img_1, fake_img_2, fake_img_3 = model(test_ref['hist'], test_ref['ab'], test_ref['l'], test_target['l'])

    # # visualize
    # visualize_results(fake_img_3)

if __name__ == '__main__':
    main()
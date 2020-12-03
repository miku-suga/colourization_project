import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from preprocess import get_tf_dataset
from model import Model

def trainMCN(model, reference_dict, target_dict, target_label_dict):
    r_hist, r_ab, r_l = r_dict["hist"], r_dict["ab"], r_dict["l"]

    t_hist, t_ab, t_l = target_dict["hist"], t_dict["ab"], t_dict["l"]
    t_label_hist, t_label_ab, t_label_l = t_label_dict["hist"], t_label_dict["ab"], t_label_dict["l"]

    num_inputs = len(train_data)
    for i in range(0, num_inputs - model.batch_size, model.batch_size):
        batch_r_hist = r_hist[i:i+model.batch_size]
        batch_r_ab = r_ab[i:i+model.batch_size]
        batch_r_l = r_l[i:i+model.batch_size]

        batch_t_hist = t_hist[i:i+model.batch_size]
        batch_t_ab = t_ab[i:i+model.batch_size]
        batch_t_l = t_l[i:i+model.batch_size]
        
        batch_t_label_l = t_label_l[i:i+model.batch_size]
        batch_t_label_ab = t_label_ab[i:i+model.batch_size]

        # must be edited layer
        with tf.GradientTape() as tape:
            g_tl, fake_img_1, fake_img_2, fake_img_3 = model(batch_r_hist, batch_r_ab, batch_r_l, batch_t_l)
            classification_loss = model.loss_class(g_tl, batch_t_label_l)
            smoothL1_loss = model.loss_pixel(batch_t_ab, batch_t_label_ab)
            hist_loss = model.loss_hist(batch_r_hist, batch_t_hist)
            tv_reg_loss = loss_tv(batch_t_ab)

            #IM STILL FIGURING OUT how to get the D value
            pass
        
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def visualise_result(model, data):
    # num_inputs = len(test_inputs)
    # loss_list = []

    # for i in range(0, num_inputs - model.batch_size, model.batch_size):
    #     batch_input = train_inputs[i:i+model.batch_size]
    #     batch_label = train_labels[i:i+model.batch_size]

    #     # probs, final_state = model(batch_input, None)
    #     # loss_list.append(model.loss(probs, batch_label))
    
    # return loss_list

def main():
    num_threads = 0
    batch_size = 48
    serial_batches = True
    no_flip = True
    display_id = -1
    
    train_data, train_label, test_data, test_label = get_tf_dataset()

    #### preprocess functions go here #####

    model = Model()
    train(model, train_data, train_label)
    loss_list = test(model, test_data, test_label)






if __name__ == '__main__':
    main()
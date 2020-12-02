import os
import tensorflow as tf
import numpy as np
import preprocess
from model import Model

def train(model, train_data, train_label):
    num_inputs = len(train_data)

    for i in range(0, num_inputs - model.batch_size, model.batch_size):
        batch_input = train_inputs[i:i+model.batch_size]
        batch_label = train_labels[i:i+model.batch_size]

        # must be edited layer
        with tf.GradientTape() as tape:
            # probs, final_state = model(batch_input, None)
            # loss = model.loss(probs, batch_label)


        # if i % 128 == 0:
        #     print(np.exp(loss))

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, data):
    num_inputs = len(test_inputs)
    loss_list = []

    for i in range(0, num_inputs - model.batch_size, model.batch_size):
        batch_input = train_inputs[i:i+model.batch_size]
        batch_label = train_labels[i:i+model.batch_size]

        # probs, final_state = model(batch_input, None)
        # loss_list.append(model.loss(probs, batch_label))
    
    return loss_list

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
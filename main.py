import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from preprocess import get_tf_dataset, get_histrogram, IMAGE_SIZE
from model import Model, Discriminator
from matplotlib import pyplot as plt
from util import calc_hist

def trainMCN(model, discrim, ref_data, target_data, noRef=False):
    i = 0
    for ref_batch, target_batch in zip(ref_data.as_numpy_iterator(), target_data.as_numpy_iterator()):
        r_l, r_ab, r_hist, _ = ref_batch
        t_l, t_ab, t_hist, t_label = target_batch

        with tf.GradientTape() as tape:
            g_tl, _, _, t_ab_out = model(
                r_hist, r_ab, r_l, t_l)

            output_img = tf.concat([t_l, t_ab_out], axis=-1)
            fake_logits = discrim(output_img)
            real_logits = discrim(tf.concat([r_l, r_ab], axis=-1))
            t_h_out = get_histrogram(t_ab_out)

            total_loss = model.loss_func(t_ab, t_ab_out, r_hist, t_h_out, fake_logits, g_tl, t_label)

        print('batch',i , 'loss=', total_loss)
        i += 1

        gradients = tape.gradient(total_loss, model.trainable_variables)
        model.optimizer.apply_gradients(
            zip(gradients, model.trainable_variables))


def visualize_results(images):
    fig = plt.figure()
    for i in range(len(images)):
        ax = fig.add_subplot(i, 1, 1)
        ax.imshow(images[i], cmap="Greys")
    plt.show()


def main():
    num_classes = 365
    batch_size = 5
    training_size = 100
    testing_size = 10

    train_target_data = get_tf_dataset(batch_size, 'train', training_size)
    train_ref_data = get_tf_dataset(
        batch_size, 'train', training_size)
    test_target_data = get_tf_dataset(batch_size, 'test', testing_size).take(1)
    test_ref_data = get_tf_dataset(batch_size, 'test', testing_size).take(1)

    model = Model(num_classes, IMAGE_SIZE, IMAGE_SIZE)
    discrim = Discriminator()

    # We are going to use target label as the train reference.
    # train MCN without histogram loss
    trainMCN(model, discrim, train_target_data, train_target_data, noRef=True)

    # train everything
    trainMCN(model, discrim, train_ref_data, train_target_data)

    # call model on first 10 test examples
    for ref_batch, target_batch in zip(test_ref_data.as_numpy_iterator(), test_target_data.as_numpy_iterator()):
        r_l, r_ab, r_hist, _ = ref_batch
        t_l, _, _, _ = target_batch
        _, _, _, fake_img_3 = model(r_hist, r_ab, r_l, t_l)

        # visualize
        visualize_results(fake_img_3)


if __name__ == '__main__':
    main()

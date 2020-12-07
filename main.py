import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import preprocess as prep
from model import Model, Discriminator


def trainMCN(model, discrim, ref_data, target_data, noRef=False):
    i = 0
    for ref_batch, target_batch in zip(ref_data.as_numpy_iterator(), target_data.as_numpy_iterator()):
        r_l, r_ab, r_hist, _ = ref_batch
        t_l, t_ab, t_hist, t_label = target_batch

        with tf.GradientTape(persistent=True) as tape:
            g_tl, t_ab_out_1, t_ab_out_2, t_ab_out_3 = model(
                r_hist, r_ab, r_l, t_l)

            output_img = tf.concat([t_l, t_ab_out_3], axis=-1)
            fake_logits = discrim(output_img)
            real_logits = discrim(tf.concat([r_l, r_ab], axis=-1))
            t_h_out_1 = prep.get_histrogram(t_ab_out_1)
            t_h_out_2 = prep.get_histrogram(t_ab_out_2)
            t_h_out_3 = prep.get_histrogram(t_ab_out_3)

            total_loss = model.loss_function(t_ab, t_ab_out_1, t_ab_out_2, t_ab_out_3,
                                             r_hist, t_h_out_1, t_h_out_2, t_h_out_3, fake_logits, g_tl, t_label, noRef)

            discrim_loss = discrim.loss_function(fake_logits, real_logits)

        tf.print('\tbatch', i, 'coloring loss =', total_loss)
        tf.print('\tbatch', i, 'discriminator loss =', discrim_loss)
        tf.print()
        i += 1

        gradients = tape.gradient(total_loss, model.trainable_variables)
        model.optimizer.apply_gradients(
            zip(gradients, model.trainable_variables))

        discrim_gradients = tape.gradient(
            discrim_loss, discrim.trainable_variables)
        discrim.optimizer.apply_gradients(
            zip(discrim_gradients, discrim.trainable_variables))


def main():
    num_classes = 365
    batch_size = 5
    training_size = 100
    testing_size = 10

    # debugging config
    # tf.config.run_functions_eagerly(True)
    # tf.debugging.enable_check_numerics()

    train_target_data = prep.get_tf_dataset(batch_size, 'train', training_size)
    train_ref_data = prep.get_tf_dataset(batch_size, 'train', training_size)
    test_target_data = prep.get_tf_dataset(
        batch_size, 'test', testing_size).take(1)
    test_ref_data = prep.get_tf_dataset(
        batch_size, 'test', testing_size).take(1)

    model = Model(num_classes, prep.IMAGE_SIZE, prep.IMAGE_SIZE)
    discrim = Discriminator()

    # We are going to use target label as the train reference.
    # train MCN without histogram loss
    trainMCN(model, discrim, train_target_data, train_target_data, noRef=True)

    # train everything
    trainMCN(model, discrim, train_ref_data, train_target_data)

    # call model on first 10 test examples
    print("Training done, visualize result..")
    for ref_batch, target_batch in zip(test_ref_data.as_numpy_iterator(), test_target_data.as_numpy_iterator()):
        r_l, r_ab, r_hist, _ = ref_batch
        t_l, _, _, _ = target_batch
        _, _, _, fake_img_3 = model(r_hist, r_ab, r_l, t_l)

        # visualize
        prep.visualize_results(t_l, fake_img_3)
        break

    # TODO: load and save the weight for GCP


if __name__ == '__main__':
    main()

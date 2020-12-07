import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import preprocess as prep
from model import Model, Discriminator


def trainMCN(model, discrim, ref_data, target_data, cp_prefix, noRef=False):
    i = 0
    loss_list = []
    for ref_batch, target_batch in zip(ref_data.as_numpy_iterator(), target_data.as_numpy_iterator()):
        r_l, r_ab, r_hist, r_label = ref_batch
        t_l, t_ab, _, t_label = target_batch

        with tf.GradientTape() as tape:
            g_tl, t_ab_out_1, t_ab_out_2, t_ab_out_3 = model(
                r_hist, r_ab, r_l, t_l)

            output_img = tf.concat([t_l, t_ab_out_3], axis=-1)
            fake_logits = discrim(output_img)
            t_h_out_1 = prep.get_histrogram(t_ab_out_1)
            t_h_out_2 = prep.get_histrogram(t_ab_out_2)
            t_h_out_3 = prep.get_histrogram(t_ab_out_3)

            total_loss = model.loss_function(t_ab, t_ab_out_1, t_ab_out_2, t_ab_out_3,
                                             r_hist, t_h_out_1, t_h_out_2, t_h_out_3, fake_logits, g_tl, t_label, noRef)

            

        with tf.GradientTape() as tape_discrim:
            fake_logits = discrim(tf.stop_gradient(output_img))
            real_logits = discrim(tf.concat([r_l, r_ab], axis=-1))
            discrim_loss = discrim.loss_function(fake_logits, real_logits)

        discrim_fake_out = tf.reduce_mean(tf.sigmoid(fake_logits))
        tf.print('\tbatch', i, 'coloring loss =', total_loss)
        tf.print('\tbatch', i, 'discriminator loss =', discrim_loss)
        tf.print('\tbatch', i, 'discriminator output =', discrim_fake_out)
        tf.print()

        loss_list.append( (total_loss, discrim_loss, discrim_fake_out) )

        i += 1

        gradients = tape.gradient(total_loss, model.trainable_variables)
        model.optimizer.apply_gradients(
            zip(gradients, model.trainable_variables))

        discrim_gradients = tape_discrim.gradient(
            discrim_loss, discrim.trainable_variables)
        discrim.optimizer.apply_gradients(
            zip(discrim_gradients, discrim.trainable_variables))

        if i % 1000 == 1:
            model.save_weights(cp_prefix + '_model')
            discrim.save_weights(cp_prefix + '_discrim')
            np.save(cp_prefix + '_loss', loss_list)

    return loss_list
def main():
    batch_size = 10
    training_size = -1
    testing_size = 100
    prefix = 'saved/'
    prefix_cp = 'checkpoints/'
    # debugging config
    # tf.config.run_functions_eagerly(True)
    # tf.debugging.enable_check_numerics()

    train_target_data = prep.get_tf_dataset(batch_size, 'train', training_size)
    train_ref_data = prep.get_tf_dataset(batch_size, 'train', training_size)
    test_target_data = prep.get_tf_dataset(
        batch_size, 'test', testing_size).take(1)
    test_ref_data = prep.get_tf_dataset(
        batch_size, 'test', testing_size).take(1)

    model = Model(prep.NUM_CLASSES, prep.IMAGE_SIZE, prep.IMAGE_SIZE)
    discrim = Discriminator()

    # We are going to use target label as the train reference.
    # train MCN without histogram loss
    loss_1 = trainMCN(model, discrim, train_target_data, train_target_data, prefix_cp + '1', noRef=True)
    np.save(prefix + 'loss_1', loss_1)
    model.save_weights(prefix + 'weights_model_1')
    discrim.save_weights(prefix + 'weights_discrim_1')

    # train everything
    loss_2 = trainMCN(model, discrim, train_ref_data, train_target_data, prefix_cp + '2')
    np.save(prefix + 'loss_2', loss_2)
    model.save_weights(prefix + 'weights_model_2')
    discrim.save_weights(prefix + 'weights_discrim_2')

    # call model on first 10 test examples
    print("Training done, visualize result..")
    for ref_batch, target_batch in zip(test_ref_data.as_numpy_iterator(), test_target_data.as_numpy_iterator()):
        r_l, r_ab, r_hist, _ = ref_batch
        t_l, _, _, _ = target_batch
        _, _, _, fake_img_3 = model(r_hist, r_ab, r_l, t_l)

        # visualize
        prep.visualize_results(t_l, fake_img_3)
        break


if __name__ == '__main__':
    main()

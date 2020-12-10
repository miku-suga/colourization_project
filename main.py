import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import preprocess as prep
from model import Model, Discriminator
import datetime


def trainMCN(model, discrim, ref_data, target_data, cp_prefix, train_log_dir, noRef=False):
    i = 0
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    for ref_batch, target_batch in zip(ref_data.as_numpy_iterator(), target_data.as_numpy_iterator()):
        r_l, r_ab, r_hist, _ = ref_batch
        t_l, t_ab, _, t_label = target_batch

        with tf.GradientTape() as tape:
            g_tl, t_ab_out_1, t_ab_out_2, t_ab_out_3 = model(
                r_hist, r_ab, r_l, t_l, noRef)

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

        gradients = tape.gradient(total_loss, model.trainable_variables)
        model.optimizer.apply_gradients(
            zip(gradients, model.trainable_variables))

        discrim_gradients = tape_discrim.gradient(
            discrim_loss, discrim.trainable_variables)
        discrim.optimizer.apply_gradients(
            zip(discrim_gradients, discrim.trainable_variables))

        # if i == 100:
        #     tf.profiler.experimental.start(train_log_dir)
        # if i == 200:
        #     tf.profiler.experimental.stop()
        if i % 1000 == 0:
            tf.print("Saving Weights..")
            model.save_weights(cp_prefix + '_model')
            discrim.save_weights(cp_prefix + '_discrim')

        with train_summary_writer.as_default():
            tf.summary.scalar('coloring_loss', total_loss, step=i)
            tf.summary.scalar('discriminator_loss', discrim_loss, step=i)
            tf.summary.scalar('discriminator_fake_output', discrim_fake_out, step=i)
        i += 1

    return
def main():
    BATCH_SIZE_1 = 10
    BATCH_SIZE_2 = 8
    TRAINING_SIZE = -1
    TESTING_SIZE = 100

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    SAVED_DIR = 'logs/saved/'
    CP_DIR = 'logs/checkpoints/'
    TRAIN_LOG_DIR = 'logs/gradient_tape/' + current_time + '/train1'
    TRAIN2_LOG_DIR = 'logs/gradient_tape/' + current_time + '/train2'
    # debugging config
    # tf.config.run_functions_eagerly(True)
    # tf.debugging.enable_check_numerics()

    model = Model(prep.NUM_CLASSES, prep.IMAGE_SIZE, prep.IMAGE_SIZE)
    discrim = Discriminator()

    # We are going to use target label as the train reference.
    # train only MCN without histogram loss
    train_target_data = prep.get_tf_dataset(BATCH_SIZE_1, 'train', TRAINING_SIZE)
    trainMCN(model, discrim, train_target_data, train_target_data, CP_DIR + '1', TRAIN_LOG_DIR, noRef=True)
    model.save_weights(SAVED_DIR + 'weights_model_1')
    discrim.save_weights(SAVED_DIR + 'weights_discrim_1')

    # train everything
    train_target_data = prep.get_tf_dataset(BATCH_SIZE_2, 'train', TRAINING_SIZE)
    train_ref_data = prep.get_tf_dataset(BATCH_SIZE_2, 'train', TRAINING_SIZE)

    # TODO: fix this
    train_ref_data = tf.image.random_flip_left_right(train_ref_data, 534234)
    train_ref_data = tf.image.random_flip_up_down(train_ref_data, 234328)

    trainMCN(model, discrim, train_ref_data, train_target_data, CP_DIR + '2', TRAIN2_LOG_DIR)
    model.save_weights(SAVED_DIR + 'weights_model_2')
    discrim.save_weights(SAVED_DIR + 'weights_discrim_2')


    # call model on first 10 test examples
    # tf.print("Training done, visualize result..")
    # test_target_data = prep.get_tf_dataset(
    #     batch_size, 'test', TESTING_SIZE)
    # test_ref_data = prep.get_tf_dataset(
    #     batch_size, 'test', TESTING_SIZE)
    # for ref_batch, target_batch in zip(test_ref_data.as_numpy_iterator(), test_target_data.as_numpy_iterator()):
    #     r_l, r_ab, r_hist, _ = ref_batch
    #     t_l, _, _, _ = target_batch
    #     _, _, _, fake_img_3 = model(r_hist, r_ab, r_l, t_l)

    #     # visualize
    #     prep.visualize_results(t_l, fake_img_3)
    #     break


if __name__ == '__main__':
    main()

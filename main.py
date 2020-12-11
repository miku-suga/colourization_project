import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import preprocess as prep
from model import Model, Discriminator
import datetime
import argparse


def trainMCN(model, discrim, ref_data, target_data, cp_prefix, train_log_dir, noRef=False):
    i = 0
    j = 0
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    for ref_batch, target_batch in zip(ref_data.as_numpy_iterator(), target_data.as_numpy_iterator()):
        r_l, r_ab, r_hist, _ = ref_batch
        t_l, t_ab, _, t_label = target_batch

        with tf.GradientTape() as tape:
            g_tl, t_ab_out_1, t_ab_out_2, t_ab_out_3 = model(
                r_hist, r_ab, r_l, t_l, noRef)

            output_img = tf.concat([t_l, t_ab_out_3], axis=-1)
            fake_logits = discrim(output_img)

            if noRef:
                t_h_out_1 = None
                t_h_out_2 = None
                t_h_out_3 = None
            else:
                t_h_out_1 = prep.get_histrogram(t_ab_out_1)
                t_h_out_2 = prep.get_histrogram(t_ab_out_2)
                t_h_out_3 = prep.get_histrogram(t_ab_out_3)

            total_loss, loss_class = model.loss_function(t_ab, t_ab_out_1, t_ab_out_2, t_ab_out_3,
                                             r_hist, t_h_out_1, t_h_out_2, t_h_out_3, fake_logits, g_tl, t_label, noRef)

        with tf.GradientTape() as tape_discrim:
            data_fake = tf.stop_gradient(output_img)
            data_real = tf.concat([r_l, r_ab], axis=-1)

            fake_logits = discrim(data_fake)
            real_logits = discrim(data_real)

            gp = discrim.gradient_penalty(data_fake, data_real)

            discrim_loss = discrim.loss_function(fake_logits, real_logits, gp)

        discrim_fake_out = tf.reduce_mean(tf.sigmoid(fake_logits))
        tf.print('\tbatch', i, 'coloring loss =', total_loss)
        tf.print('\tbatch', i, 'classification loss =', loss_class)
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

        with train_summary_writer.as_default():
            # weight checkpoint
            if i % 1000 == 999:
                tf.print("Saving Weights..")
                model.save_weights(cp_prefix + '_model')
                discrim.save_weights(cp_prefix + '_discrim')

            # output image samples
            if i % 200 == 0:
                ref_img = prep.lab2rgb_norm(r_l, r_ab)
                out_img = prep.lab2rgb_norm(tf.zeros_like(t_l), t_ab_out_3)
                
                tf.summary.image('target_image', t_l, step=j)
                tf.summary.image('reference_image', ref_img, step=j)
                tf.summary.image('output_image', out_img, step=j)
                j += 1
            
            # loss plot
            tf.summary.scalar('coloring_loss', total_loss, step=i)
            tf.summary.scalar('classification_loss', loss_class, step=i)
            tf.summary.scalar('discriminator_loss', discrim_loss, step=i)
            tf.summary.scalar('discriminator_fake_output',
                                discrim_fake_out, step=i)
        i += 1

    return


def main():
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    SAVED_DIR = 'logs/saved/' + current_time + '/'
    CP_DIR = 'logs/checkpoints/' + current_time + '/'
    TRAIN_LOG_DIR = 'logs/gradient_tape/' + current_time + '/train1'
    TRAIN2_LOG_DIR = 'logs/gradient_tape/' + current_time + '/train2'
    TEST_LOG_DIR = 'logs/gradient_tape/' + current_time + '/test'

    # command line parser
    parser = argparse.ArgumentParser(description='Train and infer color image with Gray2ColorNet')
    parser.add_argument('-m', '--model', dest='load_model_dir', help='load a saved model from the path')
    parser.add_argument('-d', '--discrim', dest='load_discrim_dir', help='load a saved discriminator from the path')
    parser.add_argument('-r', '--test', dest='run_test', help='run the test visualization')
    parser.add_argument('-t1', '--train1', dest='run_train_1', action='store_true', help='run the first training')
    parser.add_argument('-t2', '--train2', dest='run_train_2', action='store_true', help='run the second training')
    parser.add_argument('-s', '--small', dest='small', action='store_true', help='small batch size for a local debugging run')
    args = parser.parse_args()

    # debugging config
    # tf.config.run_functions_eagerly(True)
    # tf.debugging.enable_check_numerics()

    if args.small:
        BATCH_SIZE_1 = 4
        BATCH_SIZE_2 = 2
        BATCH_SIZE_TEST = 8
        TRAINING_SIZE = 100
        TESTING_SIZE = 16
    else:
        BATCH_SIZE_1 = 30
        BATCH_SIZE_2 = 8
        BATCH_SIZE_TEST = 8
        TRAINING_SIZE = -1
        TESTING_SIZE = 16

    model = Model(prep.NUM_CLASSES, prep.IMAGE_SIZE, prep.IMAGE_SIZE)
    discrim = Discriminator()

    if args.load_model_dir is not None:
        model.load_weights(args.load_model_dir)
    if args.load_discrim_dir is not None:
        discrim.load_weights(args.load_discrim_dir)

    # We are going to use target label as the train reference.
    # train only MCN without histogram loss
    if args.run_train_1:
        train_target_data = prep.get_tf_dataset(
            BATCH_SIZE_1, 'train', TRAINING_SIZE, False)
        trainMCN(model, discrim, train_target_data, train_target_data,
                CP_DIR + '1', TRAIN_LOG_DIR, noRef=True)
        model.save_weights(SAVED_DIR + 'weights_model_1')
        discrim.save_weights(SAVED_DIR + 'weights_discrim_1')

    # train everything
    if args.run_train_2:
        train_target_data = prep.get_tf_dataset(
            BATCH_SIZE_2, 'train', TRAINING_SIZE)
        train_ref_data = prep.get_tf_dataset(BATCH_SIZE_2, 'train', TRAINING_SIZE)

        # TODO: fix this
        ref_dataset = tf.data.Dataset(train_ref_data)
        for element in ref_dataset:
            element = tf.image.random_flip_left_right(element)
            element = tf.image.random_flip_up_down(element)
            element = tf.keras.preprocessing.image.random_rotation(element, 180, row_axis=0, col_axis=1, channel_axis=2)

        trainMCN(model, discrim, ref_dataset,
                train_target_data, CP_DIR + '2', TRAIN2_LOG_DIR)
        model.save_weights(SAVED_DIR + 'weights_model_2')
        discrim.save_weights(SAVED_DIR + 'weights_discrim_2')

    if args.run_test is not None:
        test_summary_writer = tf.summary.create_file_writer(TEST_LOG_DIR)
        
        # call model on first 10 test examples
        tf.print("Training done, visualize result..")
        test_target_data = prep.get_tf_dataset(
            BATCH_SIZE_TEST, 'test', TESTING_SIZE)
        test_ref_data = prep.get_tf_dataset(
            BATCH_SIZE_TEST, 'test', TESTING_SIZE)

        i = 0
        for ref_batch, target_batch in zip(test_ref_data.as_numpy_iterator(), test_target_data.as_numpy_iterator()):
            r_l, r_ab, r_hist, _ = ref_batch
            t_l, _, _, _ = target_batch
            if args.run_test == '1':
                print("Run test with MCN only")
                _, _, _, out_ab = model(r_hist, r_ab, r_l, t_l, True)
            else:
                print("Run full test")
                _, _, _, out_ab = model(r_hist, r_ab, r_l, t_l)

            # visualize
            ref_img = prep.lab2rgb_norm(r_l, r_ab)
            out_img = prep.lab2rgb_norm(t_l, out_ab)
            with test_summary_writer.as_default():
                tf.summary.image('target_image', t_l, step=i)
                tf.summary.image('reference_image', ref_img, step=i)
                tf.summary.image('output_image', out_img, step=i)

            i += 1
            
            # prep.visualize_results(t_l, fake_img_3)


if __name__ == '__main__':
    main()

import tensorflow as tf
import tensorflow_datasets as tfds
from skimage import color
import numpy as np
from PIL import Image
import cv2

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DATA_BUFFER_SIZE = 256
IMAGE_SIZE = 128
NUM_CLASSES = 32

def preprocess_dataset(img, label):
    l_ts, ab_ts = process_img(img.numpy())
    hist = get_histrogram(ab_ts)

    return (l_ts, ab_ts, hist, tf.cast(label, tf.int32))


def eager_preprocessing(data):
    return tf.py_function(preprocess_dataset, [data['image'], data['label']], (tf.float32, tf.float32, tf.float32, tf.int32))


def preprocess_filter(data):
    return data['label'] < NUM_CLASSES


'''
Returned as tensor data type
'''


def get_tf_dataset(batch_size, split, max_size=-1):

    dataset = tfds.load('places365_small', split=split, shuffle_files=True)
    dataset = dataset.filter(preprocess_filter) \
        .shuffle(DATA_BUFFER_SIZE).take(max_size) \
        .batch(batch_size).map(eager_preprocessing) \
        .prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def rgb2lab_norm(images_rgb):
    lab = color.rgb2lab(images_rgb).astype(np.float32)
    lab = tf.convert_to_tensor(lab)
    l_ts = lab[:, :, :, 0] / 50.0 - 1.0
    ab_ts = lab[:, :, :, 1:] / 110.0
    l_ts = tf.expand_dims(l_ts, axis=-1)

    return l_ts, ab_ts


def lab2rgb_norm(images_l, images_ab):
    l_ts = (images_l + 1.0) * 50.0
    ab_ts = images_ab * 110.0

    images_lab = tf.concat([l_ts, ab_ts], axis=-1)
    return color.lab2rgb(images_lab).astype(np.float32)


def process_img(images):
    images = tf.image.resize(images, (IMAGE_SIZE, IMAGE_SIZE)) / 255.0
    l_ts, ab_ts = rgb2lab_norm(images)

    return l_ts, ab_ts


def get_histrogram(ab_batch):
    img_ab = tf.transpose(ab_batch, perm=[0, 3, 1, 2])
    N, C, H, W = img_ab.shape
    assert C == 2

    img_a = tf.reshape(img_ab[:, 0, :, :], [N, 1, 1, H, W])
    img_b = tf.reshape(img_ab[:, 1, :, :], [N, 1, 1, H, W])

    grid_a = tf.reshape(tf.cast(tf.linspace(-1, 1, 21),
                                tf.float32), [1, 21, 1, 1, 1])
    grid_a = tf.broadcast_to(grid_a, [N, 21, 21, H, W])
    grid_b = tf.reshape(tf.cast(tf.linspace(-1, 1, 21),
                                tf.float32), [1, 1, 21, 1, 1])
    grid_b = tf.broadcast_to(grid_b, [N, 21, 21, H, W])

    hist_a = tf.math.maximum(0.1 - tf.abs(grid_a - img_a), 0) * 10
    hist_b = tf.math.maximum(0.1 - tf.abs(grid_b - img_b), 0) * 10
    hist = tf.reshape(tf.reduce_mean(hist_a * hist_b, axis=(3, 4)), [N, -1])
    assert hist.shape == (N, 441)

    return hist


def visualize_results(images_l, images_ab):
    images = lab2rgb_norm(images_l, images_ab)
    # images reshape to (batch_size, D)
    images = np.reshape(images, [images.shape[0], -1, 3])
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg, sqrtimg, 3]))

    plt.show()


if __name__ == "__main__":
    batch_size = 1000
    max_size = 10

    # find min and max value in LAB color space
    range1 = tf.broadcast_to(tf.reshape(
        tf.range(256), [1, 1, 256, 1]), [256, 256, 256, 1])
    range2 = tf.broadcast_to(tf.reshape(
        tf.range(256), [1, 256, 1, 1]), [256, 256, 256, 1])
    range3 = tf.broadcast_to(tf.reshape(
        tf.range(256), [256, 1, 1, 1]), [256, 256, 256, 1])

    test_img = tf.cast(
        tf.concat([range1, range2, range3], axis=-1), tf.float32) / 255.0
    print(test_img.shape)
    test_l, test_ab = rgb2lab_norm(test_img)
    print("L min", tf.reduce_min(
        test_l[:, :, :, 0]), "max", tf.reduce_max(test_l[:, :, :, 0]))
    print("A min", tf.reduce_min(
        test_ab[:, :, :, 0]), "max", tf.reduce_max(test_ab[:, :, :, 0]))
    print("B min", tf.reduce_min(
        test_ab[:, :, :, 1]), "max", tf.reduce_max(test_ab[:, :, :, 1]))

    # load data
    train_data = get_tf_dataset(batch_size, 'train')
    # test_data = get_tf_dataset(batch_size, 'test', max_size)

    print("dataset loaded")
    count = 0
    for imgs in train_data.as_numpy_iterator():
        count += len(imgs)

        print (count)

    # TODO: randomly pair them with ref

    # ## set refs to the original colored image
    # train_ref = train_data[:]
    # test_ref = test_data[:]

    # ## set input images to the greyscaled images
    # train_target = []
    # test_target = []
    # for i in train_data:
    #     train_target.append(rgb2gray(i))
    # for i in test_data:
    #     test_target.append(rgb2gray(i))

    # ## put input and label into list of dictionaries containing processed info
    # train_dict_1, train_dict_2 = create_dict(train_target, train_ref)
    # test_dict_1, test_dict_2 = create_dict(test_target, test_ref)

    # print(len(train_dict_1))

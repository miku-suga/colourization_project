import tensorflow as tf
import tensorflow_datasets as tfds
from skimage import color
import numpy as np
from PIL import Image
import cv2

DATA_BUFFER_SIZE = 256
IMAGE_SIZE = 64


def preprocess_dataset(img, label):
    l_ts, ab_ts = process_img(img.numpy())
    hist = get_histrogram(ab_ts)

    return (l_ts, ab_ts, hist, label)


def eager_preprocessing(data):
    return tf.py_function(preprocess_dataset, [data['image'], data['label']], (tf.float32, tf.float32, tf.float32, tf.int64))


'''
Returned as tensor data type
'''


def get_tf_dataset(batch_size, split, max_size=-1):

    dataset = tfds.load('places365_small', split=split, shuffle_files=True)
    dataset = dataset.shuffle(DATA_BUFFER_SIZE).take(
        max_size).batch(batch_size).map(eager_preprocessing)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # TODO: filter to only 50 labels?

    return dataset


def rgb2gray(img):
    return np.dot(img[..., :3], [0.299, 0.587, 0.114])


def process_img(images):
    images = tf.image.resize(images, (IMAGE_SIZE, IMAGE_SIZE))
    lab = color.rgb2lab(images).astype(np.float32)
    lab = tf.convert_to_tensor(lab)

    l_ts = lab[:, :, :, 0] / 50.0 - 1.0
    ab_ts = lab[:, :, :, 1:] / 110.0

    l_ts = tf.expand_dims(l_ts, axis=-1)
    return l_ts, ab_ts


def get_histrogram(ab_batch):
    img_ab = tf.transpose(ab_batch, perm=[0, 3, 1, 2])
    N, C, H, W = img_ab.shape
    assert C == 2

    img_a = tf.reshape(img_ab[:, 0, :, :], [N, 1, 1, H, W])
    img_b = tf.reshape(img_ab[:, 1, :, :], [N, 1, 1, H, W])

    grid_a = tf.reshape(tf.cast(tf.linspace(-1, 1, 21), tf.float32), [1, 21, 1, 1, 1])
    grid_a = tf.broadcast_to(grid_a, [N, 21, 21, H, W])
    grid_b = tf.reshape(tf.cast(tf.linspace(-1, 1, 21), tf.float32), [1, 1, 21, 1, 1])
    grid_b = tf.broadcast_to(grid_b, [N, 21, 21, H, W])

    hist_a = tf.math.maximum(0.1 - tf.abs(grid_a - img_a), 0) * 10
    hist_b = tf.math.maximum(0.1 - tf.abs(grid_b - img_b), 0) * 10
    hist = tf.reshape(tf.reduce_mean(hist_a * hist_b, axis=(3, 4)), [N, -1] )
    assert hist.shape == (N, 441)

    return hist


def get_target(dataset):
    target_dataset = []
    for i in dataset:
        target_dataset.append(rgb2gray(i))
    return target_dataset


if __name__ == "__main__":
    batch_size = 5
    max_size = 10

    # load data
    train_data = get_tf_dataset(batch_size, 'train', max_size)
    test_data = get_tf_dataset(batch_size, 'test', max_size)

    print("dataset loaded")
    for img in train_data.as_numpy_iterator():
        i_l, i_ab, i_hist, i_label = img
        # shape
        # i_l = (batch_size, IMAGE_SIZE, IMAGE_SIZE, 1)
        # i_ab = (batch_size, IMAGE_SIZE, IMAGE_SIZE, 2)
        # i_label = (batch_size)
        print(i_l, i_ab, i_hist, i_label)

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

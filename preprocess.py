import tensorflow as tf
import tensorflow_datasets as tfds
from skimage import color
import numpy as np
from PIL import Image
import cv2

'''
Returned as tensor data type
'''
def get_tf_dataset():
    train_data, train_label = tfds.load('places365_small', split='train', batch_size=-1, as_supervised=True)
    test_data, test_label = tfds.load('places365_small', split='test', batch_size=-1, as_supervised=True)

    # Comment this part out if we're using the full dataset:
    train_data = train_data[:60000]
    train_label = train_label[:60000]
    test_data = test_data[:10000]
    test_label = test_label[:10000]

    return train_data, train_label, test_data, test_label


def process_img(im):
    im = np.array(im)
    ims = [im]
    for i in [0.5, 0.25]:
        ims = [cv2.resize(im, None, fx=i, fy=i, interpolation=cv2.INTER_AREA)] + ims
    l_ts, ab_ts = [], []
    for im in ims:
        lab = color.rgb2lab(im).astype(np.float32)
        lab_t = tf.convert_to_tensor(lab)
        l_ts.append(lab_t[[0], ...] / 50.0 - 1.0)
        ab_ts.append(lab_t[[1, 2], ...] / 110.0)
    return l_ts, ab_ts

def create_dicts(image_train, label_train, image_test, label_test):
    image_train, label_train, image_test, label_test = get_tf_dataset()
    print ("image loaded")
    image_list = []
    label_list = []
    for i in range (len(image_train)):
        image = image_train[i]
        label = label_train[i]
        tf.image.resize(image, (64,64))
        tf.image.resize(label, (64,64))
        l, ab = process_img(image)
        label_l, label_ab = process_img(label)
        im_dict = {
            'l' : l,
            'ab' : ab,
            'hist' : ab,
        }
        l_dict = {
            'l' : label_l,
            'ab' : label_ab,
            'hist' : label_ab,
        }
        image_list.append(im_dict) 
        label_list.append(l_dict)

    return image_list, label_list

if __name__ == "__main__":
    image_train, label_train, image_test, label_test = get_tf_dataset()
    print ("image loaded")
    image_list = []
    label_list = []
    for i in range (len(image_train)):
        image = image_train[i]
        label = label_train[i]
        tf.image.resize(image, (64,64))
        tf.image.resize(label, (64,64))
        l, ab = process_img(image)
        label_l, label_ab = process_img(label)
        im_dict = {
            'l' : l,
            'ab' : ab,
            'hist' : ab,
        }
        l_dict = {
            'l' : label_l,
            'ab' : label_ab,
            'hist' : label_ab,
        }
        image_list.append(im_dict) 
        label_list.append(l_dict)

    print (len(image_list))
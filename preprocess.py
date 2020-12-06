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
    return (l_ts, ab_ts, label)

def eager_preprocessing(data):
  return tf.py_function(preprocess_dataset, [data['image'], data['label']], (tf.float32, tf.float32, tf.int64))

'''
Returned as tensor data type
'''
def get_tf_dataset(batch_size, split, max_size=-1):
    
    dataset = tfds.load('places365_small', split=split, shuffle_files=True)
    dataset = dataset.shuffle(DATA_BUFFER_SIZE).take(max_size).map(eager_preprocessing)
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

def rgb2gray(img):
    return np.dot(img[...,:3], [0.299, 0.587, 0.114])

def process_img(im):
    im = cv2.resize(im, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

    lab = color.rgb2lab(im).astype(np.float32)
    lab = tf.convert_to_tensor(lab)

    l_ts = lab[:, :, 0] / 50.0 - 1.0
    ab_ts = lab[:, :, 1:] / 110.0

    return l_ts, ab_ts

def create_dict(refs, targets):
    t_l_list, t_ab_list, t_hist_list = [], [], []
    r_l_list, r_ab_list, r_hist_list = [], [], []

    for i in range (len(targets)):
        target = targets[i]
        reference = refs[i]
        t_l, t_ab = process_img(target)
        r_l, r_ab = process_img(reference)

        # TODO: fix this
        t_hist = 1
        r_hist = 1

        t_l_list.append(t_l)
        t_ab_list.append(t_ab)
        t_hist_list.append(t_hist)
        r_l_list.append(r_l)
        r_ab_list.append(r_ab)
        r_hist_list.append(r_hist)
    
    ref_dict = {
        'r_l' : r_l_list,
        'r_ab' : r_ab_list,
        'r_hist' : r_hist_list
    }

    target_dict = {
        't_l' : t_l_list,
        't_ab' : t_ab_list,
        't_hist' : t_hist_list
    }
        
    return ref_dict, target_dict

def get_target(dataset):
    target_dataset = []
    for i in dataset:
        target_dataset.append(rgb2gray(i))
    return target_dataset



if __name__ == "__main__":
    batch_size = 5
    max_size= 10

    ## load data
    train_data = get_tf_dataset(batch_size, 'train', max_size)
    test_data = get_tf_dataset(batch_size, 'test', max_size)

    print("dataset loaded")
    for img in train_data.as_numpy_iterator():
        i_l, i_ab, i_label = img
        # shape
        # i_l = (batch_size, IMAGE_SIZE, IMAGE_SIZE)
        # i_ab = (batch_size, IMAGE_SIZE, IMAGE_SIZE, 2)
        # i_label = (batch_size)

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
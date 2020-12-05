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
    # train_data, _ = tfds.load('places365_small', split='train', batch_size=-1, as_supervised=True)
    # test_data, _ = tfds.load('places365_small', split='test', batch_size=-1, as_supervised=True)
    train_data, _ = tfds.load('places365_small', split='train[:100]', batch_size=-1, as_supervised=True)
    test_data, _ = tfds.load('places365_small', split='train[:100]', batch_size=-1, as_supervised=True)

    print(tf.shape(train_data))

    train_data = tfds.as_numpy(train_data)
    test_data = tfds.as_numpy(test_data)

    print(tf.shape(train_data))
    # test_data = tfds.as_numpy(test_data)

    return train_data, test_data

def rgb2gray(img):
    return np.dot(img[...,:3], [0.299, 0.587, 0.114])

def process_img(im):
    im = np.array(im)
    ims = [im]
    for i in [0.5, 0.25]:
        ims = [cv2.resize(im, None, fx=i, fy=i, interpolation=cv2.INTER_AREA)] + ims
    l_ts, ab_ts = [], []
    for im in ims:
        lab = color.rgb2lab(im).astype(np.float32)
        lab_t = tf.convert_to_tensor(lab)
        l_ts.append(lab_t[0] / 50.0 - 1.0)
        ab_ts.append(lab_t[1:2] / 110.0)
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
    ## load data
    train_data, test_data = get_tf_dataset()
    print("dataset loaded")

    ## resize images
    for img in train_data:
        img = cv2.resize(img, (64, 64))
    for img in test_data:
        img = cv2.resize(img, (64, 64))
    # train_data = cv2.resize(train_data, (64, 64))
    # test_data = cv2.resize(test_data, (64, 64))

    ## set refs to the original colored image
    train_ref = train_data[:]
    test_ref = test_data[:]

    ## set input images to the greyscaled images
    train_target = []
    test_target = []
    for i in train_data:
        train_target.append(rgb2gray(i))
    for i in test_data:
        test_target.append(rgb2gray(i))

    ## put input and label into list of dictionaries containing processed info
    train_dict_1, train_dict_2 = create_dict(train_target, train_ref)
    test_dict_1, test_dict_2 = create_dict(test_target, test_ref)

    print(len(train_dict_1))
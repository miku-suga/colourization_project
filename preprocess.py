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

    print (tf.shape(train_data))

    train_data = tfds.as_numpy(train_data)
    test_data = tfds.as_numpy(test_data)

    print (tf.shape(train_data))
    # test_data = tfds.as_numpy(test_data)

    # Comment this part out if we're using the full dataset:
    # train_data = train_data[:60000]
    # test_data = test_data[:10000]

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

def create_dict(targets, refs):
    target_list = []
    ref_list = []
    for i in range (len(targets)):
        target = targets[i]
        reference = refs[i]
        t_l, t_ab = process_img(target)
        r_l, r_ab = process_img(reference)
        target_dict = {
            't_l' : t_l,
            't_ab' : t_ab,
            't_hist' : t_ab,
        }
        ref_dict = {
            'r_l' : r_l,
            'r_ab' : r_ab,
            'r_hist' : r_ab
        }
        target_list.append(target_dict)
        ref_list.append(ref_dict)
        
    return target_list, ref_list

if __name__ == "__main__":
    ## load data
    train_data, test_data = get_tf_dataset()
    print ("dataset loaded")

    ## resize images
    train_data = cv2.resize(train_data, (64, 64))
    test_data = cv2.resize(test_data, (64, 64))

    ## set refs to the original colored image
    train_ref = train_data
    test_ref = test_data

    ## set input images to the greyscaled images
    train_target= []
    test_target = []
    for i in train_data:
        train_target.append(rgb2gray(i))
    for i in test_data:
        test_target.append(rgb2gray(i))

    ## put input and label into list of dictionaries containing processed info
    train_dict_1, train_dict_2 = create_dict(train_target, train_ref)
    test_dict_1, test_dict_2 = create_dict(test_target, test_ref)

    print (len(train_dict_1))
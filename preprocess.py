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
    train_data, _ = tfds.load('places365_small', split='train[:1%]', batch_size=-1, as_supervised=True)
    train_data, _ = tfds.load('places365_small', split='train[:1%]', batch_size=-1, as_supervised=True)

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
        l_ts.append(lab_t[[0], ...] / 50.0 - 1.0)
        ab_ts.append(lab_t[[1, 2], ...] / 110.0)
    return l_ts, ab_ts

if __name__ == "__main__":
    ## load data
    train_data, test_data = get_tf_dataset()
    print ("dataset loaded")

    ## set labels to the original colored image
    train_label = train_data
    test_label = test_data

    ## set input images to the greyscaled images
    train_input= []
    test_input = []
    for i in train_data:
        train_input.append(rgb2gray(i))
    for i in test_data:
        test_input.append(rgb2gray(i))

    ## put input and label into list of dictionaries containing processed info
    input_list = []
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
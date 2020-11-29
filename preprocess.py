import tensorflow as tf
from skimage import color
import numpy as np
from PIL import Image
import cv2

def lab2rgb(self, L, AB):
    AB2 = AB * 110.0
    L2 = (L + 1.0) * 50.0
    Lab = tf.concat([L2, AB2], dim=1)
    Lab = Lab[0].data.cpu().float().numpy()
    Lab = np.transpose(Lab.astype(np.float64), (1, 2, 0))
    rgb = color.lab2rgb(Lab) * 255
    return rgb

def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if ow <= oh:
        if (ow == target_width):
            return img
        w = target_width
        h = int(target_width * oh / ow)
    else:
        if (oh == target_width):
            return img
        h = target_width
        w = int(target_width * ow / oh)
    return img.resize((w, h), method)

def process_img(self, im_path, transform):
    im = Image.open(im_path).convert('RGB')
    im = transform(im)
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


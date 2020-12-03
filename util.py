import numpy as np
import tensorflow as tf
from skimage import color

def calc_hist(data_ab, device):
    N, C, H, W = data_ab.shape
    grid_a = tf.linspace(-1, 1, 21).view(1, 21, 1, 1, 1).expand(N, 21, 21, H, W).to(device)
    grid_b = tf.linspace(-1, 1, 21).view(1, 1, 21, 1, 1).expand(N, 21, 21, H, W).to(device)
    hist_a = tf.math.maximum(0.1 - tf.abs(grid_a - data_ab[:, 0, :, :].view(N, 1, 1, H, W)), tf.Tensor([0]).to(device)) * 10
    hist_b = tf.math.maximum(0.1 - tf.abs(grid_b - data_ab[:, 1, :, :].view(N, 1, 1, H, W)), tf.Tensor([0]).to(device)) * 10
    hist = (hist_a * hist_b).mean(dim=(3, 4)).view(N, -1)
    return hist

def lab2rgb(L, AB):
    AB2 = AB * 110.0
    L2 = (L + 1.0) * 50.0
    Lab = tf.concat([L2, AB2], 1)
    Lab = Lab[0].data.cpu().float().numpy()
    Lab = np.transpose(Lab.astype(np.float64), (1, 2, 0))
    rgb = color.lab2rgb(Lab) * 255
    return rgb

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, tf.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)
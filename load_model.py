from __future__ import print_function
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

from keras.applications import vgg19
from keras import backend as K

from keras.preprocessing.image import load_img, save_img, img_to_array
from scipy.optimize import fmin_l_bfgs_b
import time
import argparse

new_model = load_model('42_last.h5')

base_image_path = 'input.png'
width, height = load_img(base_image_path).size
img_nrows = 400
img_ncols = int(width * img_nrows / height)

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, img_nrows, img_ncols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

for i in range (0, 1):
    # load an image in PIL format
    original = load_img(base_image_path)

    x = preprocess_image(base_image_path)
    # result = new_model.predict(original.copy())
    img = deprocess_image(x)
    save_img("AAA.png", img)
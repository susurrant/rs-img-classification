

from __future__ import division

import os
from keras import backend as K
from keras.backend import binary_crossentropy
import numpy as np
import re

K.set_image_dim_ordering('tf')
smooth = 1e-12


def jaccard_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_loss(y_true, y_pred):
    return -K.log(jaccard_coef(y_true, y_pred)) + binary_crossentropy(y_pred, y_true)


def binary_accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / y_true.size


def binary_recall(y_true, y_pred):
    idx = np.where(y_true == 1)
    if y_true[idx].size == 0:
        return -1
    else:
        return np.sum(y_true[idx] == y_pred[idx]) / y_true[idx].size


def binary_precision(y_true, y_pred):
    idx = np.where(y_pred == 1)
    if y_pred[idx].size == 0:
        return -1
    else:
        return np.sum(y_true[idx] == y_pred[idx]) / y_pred[idx].size


def search_best_model(path):
    pattern = re.compile(r'(?<=-)(\d+\.\d+)(?=\.)')
    files = os.listdir(path)
    loss = float('inf')
    model_file = ''
    for fn in files:
        tem_loss = float(pattern.search(fn).group())
        if tem_loss < loss:
            loss = tem_loss
            model_file = fn
    return os.path.join(path, model_file)


def predict(model, image):
    """ Predict mask of image

    :param image: image in numpy array with size (H, W, Channels)
    :return: list of binary masks in numpy array format (Categories, H, W)
    """
    res = _bin_mask(model.predict(image[np.newaxis, ...]))
    mask = np.squeeze(res)

    return mask


def _bin_mask(image):
    """ Clip the image into binary image
    When pixel in [0, 0.5) => 0, else => 1

    :param image: image numpy array
    :return: binary image
    """
    return np.clip(image, 0, 1) >= 0.5
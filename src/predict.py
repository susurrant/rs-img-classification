
import os
import numpy as np
from keras.models import load_model
from keras import backend as K
from keras.backend import binary_crossentropy
import tifffile as tif
from skimage import img_as_ubyte
import argparse

smooth = 1e-12
K.set_image_dim_ordering('tf')

img_rows = 1024
img_cols = 1024

num_channels = 3
num_mask_channels = 1


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--category', help='object type', type=str, default='Building')
    parser.add_argument('-m', '--model', help='model file', type=str, default='weights.03-2.06.hdf5')
    args = parser.parse_args()

    weight_path = "../checkpoints/%s" % args.category
    model = load_model(
        os.path.join(weight_path, args.model),
        custom_objects={
            u'jaccard_coef_loss': jaccard_coef_loss,
            u'jaccard_coef_int': jaccard_coef_int
        })

    img_path = '../data/' + args.category
    img = tif.imread('../0.tif').astype(np.float16)
    for c in range(num_channels):
        img[:, :, c] = (img[:, :, c] - img[:, :, c].min()) / (img[:, :, c].max() - img[:, :, c].min())

    gt = img_as_ubyte(tif.imread(gt_path))  # with regard to the type of gt img
    gt = _convert_mask(gt, RGB)

    model.predict(img)
import numpy as np
from keras.models import Model
from keras.models import load_model
from keras import backend as K
from keras.backend import binary_crossentropy
import tifffile as tif

smooth = 1e-12
K.set_image_dim_ordering('tf')

img_rows = 112
img_cols = 112

num_channels = 4
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
    model = load_model(
        '../wights/0_Building',
        custom_objects={
            u'jaccard_coef_loss': jaccard_coef_loss,
            u'jaccard_coef_int': jaccard_coef_int
        })

    img = tif.imread('../0.tif').astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min())
    model.predict(img)
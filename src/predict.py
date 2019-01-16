
import os
import numpy as np
from keras.models import load_model
from keras import backend as K
from keras.backend import binary_crossentropy
import tifffile as tif
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='image path', type=str, default='')
    parser.add_argument('-i', '--image', help='image file', type=str, default='')
    args = parser.parse_args()

    categories = ['Airport', 'Baresoil', 'Building', 'Farmland', 'Road', 'Vegetation', 'Water']
    models = []

    if args.path:
        if args.image:  # predict one image using all models
            for i, cate in enumerate(categories):
                weight_path = "../checkpoints/%s" % cate
                model = load_model(
                    os.path.join(weight_path, models[i]),
                    custom_objects={
                        u'jaccard_coef_loss': jaccard_coef_loss,
                        u'jaccard_coef_int': jaccard_coef_int
                    })

                img = tif.imread(os.path.join(args.path, args.image)).astype(np.float16)
                for c in range(num_channels):
                    img[:, :, c] = (img[:, :, c] - img[:, :, c].min()) / (img[:, :, c].max() - img[:, :, c].min())
                mask_pred = predict(model, img)
                tif.imsave(os.path.join('../predict', args.image[:-3]+'_'+cate+'tif'), mask_pred)
        else:
            print 'Please input an image name.'  # can be modified to predict all images in the directory args.path
    else:
        print 'Please input an image path.'
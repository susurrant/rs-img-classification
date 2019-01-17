
from __future__ import division

import os
import numpy as np
from keras.models import load_model
from keras import backend as K
import tifffile as tif
import argparse

from func import *

K.set_image_dim_ordering('tf')

num_channels = 3
num_mask_channels = 1


def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', help='image path', type=str, default='')
    parser.add_argument('-i', '--image', help='image file', type=str, default='')

    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parser()

    categories = ['Airport', 'Baresoil', 'Building', 'Farmland', 'Road', 'Vegetation', 'Water']

    if args.path:
        if args.image:  # predict one image using all models
            for cate in categories:
                weight_path = "../checkpoints/%s" % cate
                model = load_model(
                    search_best_model(weight_path),
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
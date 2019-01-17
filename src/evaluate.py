
from __future__ import division

import os
import numpy as np
from keras.models import load_model
from keras import backend as K
import tifffile as tif
from skimage import img_as_ubyte
import argparse
import tqdm

from func import *

K.set_image_dim_ordering('tf')

num_channels = 3
num_mask_channels = 1


def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--category', help='object type', type=str, default='Building')
    parser.add_argument('-m', '--model', help='model file', type=str, default='')

    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parser()

    weight_path = "../checkpoints/%s" % args.category

    if args.model:
        model_file = os.path.join(weight_path, args.model)
    else:
        model_file = search_best_model(weight_path)

    model = load_model(
        model_file,
        custom_objects={
            u'jaccard_coef_loss': jaccard_coef_loss,
            u'jaccard_coef_int': jaccard_coef_int
        })

    br = 0
    bp = 0
    l = 0
    img_path = '../data/' + args.category
    test_idx = np.loadtxt(os.path.join(img_path, 'test.txt'), dtype=np.uint16, delimiter=' ')
    for idx in test_idx:
        img = tif.imread(os.path.join(img_path, '%d.tif' % idx)).astype(np.float16)
        for c in range(num_channels):
            img[:, :, c] = (img[:, :, c] - img[:, :, c].min()) / (img[:, :, c].max() - img[:, :, c].min())

        mask_pred = predict(model, img)
        mask = img_as_ubyte(tif.imread(os.path.join(img_path, '%d_mask.tif' % idx))).astype(np.float16)[12:1012, 12:1012]  # with regard to the output size
        r = binary_recall(mask, mask_pred)
        p = binary_precision(mask, mask_pred)
        if r != -1 and p != -1:
            br += r
            bp += p
            l += 1
            print idx, r, p

    print 'binary precision for test data:', bp/l
    print 'binary recall for test data:', br/l
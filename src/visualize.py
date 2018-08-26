from __future__ import division
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.transform as transform
import tifffile as tif
import pandas as pd

from keras.models import load_model
from keras.backend import binary_crossentropy
from keras import backend as K

os.system('export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64')
os.environ['KERAS_BACKEND']='tensorflow'
os.environ['PATH']='/usr/local/cuda-8.0/bin'
os.environ['LD_LIBRARY_PATH']='/usr/local/cuda-8.0/lib64'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

K.set_image_dim_ordering('tf')

IMAGE_SIZE = 112
MASK_SIZE = 80
WEIGHT_PATH = [
    "../checkpoints/0Building/weights.67-1.65.hdf5",
    "../checkpoints/1Tree/weights.91-3.15.hdf5",
    "../checkpoints/2Grass/weights.36-0.93.hdf5",
    "../checkpoints/3Unknown/weights.09-0.00.hdf5",
    "../checkpoints/4Car/weights.99-0.31.hdf5",
    "../checkpoints/5Road/weights.74-1.80.hdf5"
]
IMAGE_PATH = '../data/image_canada/top_potsdam_6_12_RGBIR.tif'
MASK_PATH = '../data/mask_canada/top_potsdam_6_12_label.tif'
MASK_DIR = '../data/mask_canada/'
OUTPUT_PATH = '../predictions/'

smooth = 1e-12


class Model():
    """ unet model
    """

    def __init__(self, weight_paths=None):
        """ Load weights from every category
        Form len(category) binary classifiers

        :param wegith_paths: list of weight path of weights from every weight path
        """
        self.weight_paths = weight_paths
        self.custom_objects = {
            u'jaccard_coef_loss': self._jaccard_coef_loss,
            u'jaccard_coef_int': self._jaccard_coef_int
        }
        self.category_code = {
            0: [0, 0, 255],
            1: [0, 255, 0],
            2: [0, 255, 255],
            3: [255, 0, 0],
            4: [255, 255, 0],
            5: [255, 255, 255],
        }
        # every binary classifier for categories
        self.models = [load_model(path, custom_objects=self.custom_objects) for path in self.weight_paths]
        print " ===> model loaded"

    def predict(self, image):
        """ Predict mask of image

        :param image: image in numpy array with size (H, W, Channels)
        :return: list of binary masks in numpy array format (Categories, H, W)
        """
        height, width = image.shape[:2]
        masks = np.zeros(shape=(height, width, len(self.models)))
        image = self._normalize_image(image)  # normalization
        image = np.pad(image, ((16, 16), (16, 16), (0, 0)), 'constant', constant_values=0)  # padding

        pad_size_half = int(IMAGE_SIZE / 2)
        mask_size_half = int(MASK_SIZE / 2)
        overlap_size = pad_size_half - mask_size_half

        for i in tqdm(range(pad_size_half, height, MASK_SIZE)):
            for j in range(pad_size_half, width, MASK_SIZE):
                y1_image = i - pad_size_half
                y2_image = i + pad_size_half
                x1_image = j - pad_size_half
                x2_image = j + pad_size_half

                image_clip = image[y1_image:y2_image, x1_image:x2_image, :]
                res = [self._bin_mask(model.predict(image_clip[np.newaxis, ...])) for model in self.models]

                y1_mask = i - mask_size_half - overlap_size
                y2_mask = i + mask_size_half - overlap_size
                x1_mask = j - mask_size_half - overlap_size
                x2_mask = j + mask_size_half - overlap_size

                for k in range(len(self.models)):
                    masks[y1_mask:y2_mask, x1_mask:x2_mask, k] = np.squeeze(res[k])

        return masks

    def predict_visualize(self, image):
        """ Get the predicted masks and visualize them in Jupyter Notebook
        Image and masks all will be visualized

        :param image: raw image
        """
        masks = self.predict(image)
        plt.figure(figsize=(16, 16))
        plt.title("Image")
        plt.imshow(image[..., :3])
        plt.show()

        plt.figure(figsize=(16, 16))
        for i in range(len(self.weight_paths)):
            plt.subplot(2, 3, i + 1)
            plt.title(self.weight_paths[i])
            plt.imshow(masks[..., i])
        plt.show()

    def predict_visualize_RGB(self, image, mask_gt=None, masks_pred = None, order=(3, 5, 0, 1, 2, 4), jaccard=False, save_path = None):
        """ Viusalize the raw image, ground truth and predicted mask
        Mask overlap order: Unknown, Road, Building, Grass, Tree, Car

        :param image: raw image in numpy (N, H, W, C)
        :param mask: mask ground truth in numpy (N, H, W, C)
        """
        if masks_pred is None:
            masks_pred = self.predict(image)
        height, width = masks_pred.shape[:2]
        mask_pred = np.zeros(shape=(height, width, 3), dtype=np.uint8)
        # use category color code to colorate the mask_pred
        for i in order:
            mask_category = masks_pred[..., i]
            colors = self.category_code[i]
            for channel, color in enumerate(colors):
                mask_pred[..., channel][mask_category.astype(np.bool)] = color

        # plot image, mask_gt, mask_pred
        plt.figure(figsize=(16, 16))

        # plot image
        if mask_gt is None:
            plt.subplot(1, 2, 1)
        else:
            plt.subplot(1, 3, 1)
        plt.title("Image")
        plt.imshow(image[..., :3])

        # plot mask_pred
        if mask_gt is None:
            plt.subplot(1, 2, 2)
        else:
            plt.subplot(1, 3, 2)
        plt.title("Mask_pred")
        plt.imshow(mask_pred)

        # plot mask_gt
        if mask_gt is None:
            plt.show()
            return
        else:
            plt.subplot(1, 3, 3)
            plt.title("Mask_gt")
            plt.imshow(mask_gt)

        # show all
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)

        jaccards = []
        # compute jaccard index
        if mask_gt is not None and jaccard:
            masks_pred = self._rgb2masks(mask_pred)
            masks_gt = self._rgb2masks(mask_gt)

            for i in range(len(self.category_code)):
                jaccards.append(self._jaccard_index(masks_gt[..., i], masks_pred[..., i]))
            return jaccards

    def _rgb2masks(self, mask):
        height, width = mask.shape[:2]
        masks = np.zeros(shape=(height, width, len(self.category_code)), dtype=np.bool)
        for i in range(len(self.category_code)):
            R, G, B = self.category_code[i]
            masks[..., i] = (mask[..., 0] == R) & (mask[..., 1] == G) & (mask[..., 2] == B)
        return masks

    def _normalize_image(self, image):
        """ Normalize the input image

        :param image: image in numpy array
        :return: normalized image with the same size
        :rtype: numpy array
        """
        image = image.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min())
        return image

    def _bin_mask(self, image):
        """ Clip the image into binary image
        When pixel in [0, 0.5) => 0, else => 1

        :param image: image numpy array
        :return: binary image
        """
        return np.round(np.clip(image, 0, 1)).astype(np.bool)

    """ Jaccard loss and metric
    Credicted to http://blog.kaggle.com/2017/05/09/dstl-satellite
    -imagery-competition-3rd-place-winners-interview-vladimir-sergey/
    """

    def _jaccard_coef(self, y_true, y_pred):
        intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
        sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

        jac = (intersection + smooth) / (sum_ - intersection + smooth)

        return K.mean(jac)

    def _jaccard_coef_int(self, y_true, y_pred):
        y_pred_pos = K.round(K.clip(y_pred, 0, 1))

        intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
        sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])

        jac = (intersection + smooth) / (sum_ - intersection + smooth)

        return K.mean(jac)

    def _jaccard_coef_loss(self, y_true, y_pred):
        return -K.log(self._jaccard_coef(y_true, y_pred)) + binary_crossentropy(y_pred, y_true)

    def _jaccard_index(self, y_true, y_pred):
        intersection = (y_true & y_pred).sum()
        sum_ = (y_true | y_pred).sum()

        jaccard = (intersection + smooth) / (sum_ + smooth)

        return jaccard


if __name__ == "__main__":
    model = Model(WEIGHT_PATH)

    image = tif.imread(IMAGE_PATH)
    mask = tif.imread(MASK_PATH)
    a = model.predict_visualize_RGB(image, mask, save_path='../predictions/1,jpg')

    df = pd.DataFrame(a)
    df.insert(loc=0, data=[IMAGE_PATH], column='path')
    df.to_csv('../predictions/test.csv')

    # mask_paths = os.listdir(MASK_DIR)
    # for mask_path in mask_paths:
    #     if mask_path.find('.tif') == -1:
    #         del mask_path
    # length = int(len(mask_paths) / 2)
    # mask_paths = mask_paths[:length]
    # for mask_path in mask_paths:
    #     mask = tif.imread(mask_path)
    #     image = tif.imread(mask_path.replace('label', 'RGBIR'))

        
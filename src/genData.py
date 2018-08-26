import cv2
import numpy as np
import tifffile

path = '../data/VAZ1_201709291702_001_0023_L1A/'
tif = tifffile.imread(path + 'VAZ1_201709291702_001_0023_L1A.tif')
tif_gray = cv2.imread(path + 'VAZ1_201709291702_001_0023_L1A.tif', cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(path + 'VAZ1_201709291702_001_0023_L1A_mask.tif')
mask = cv2.resize(mask, tif_gray.shape[::-1])
print(tif.shape, tif_gray.shape, mask.shape)

width, height = tif_gray.shape[:2]

col_step = 50
row_step = 50
size = 1024
img_id = 0

for j in range(0, width, col_step):
    if j%500 == 0:
        print('rows:', j)
    for i in range(0, height, row_step):
        ttg = tif_gray[j:j+size,i:i+size]
        if np.sum(ttg==0) >= 5:
            continue
        tt = tif[j:j+size, i:i+size]
        tm = mask[j:j+size,i:i+size]
        tifffile.imsave(path + 'data/' + str(img_id) + '.tif', tt)
        tifffile.imsave(path + 'data/' + str(img_id) + '_mask.tif', tm)
        img_id += 1



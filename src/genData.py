

import os
import numpy as np
import tifffile

obj_types = ['Airport', 'Baresoil', 'Building', 'Farmland', 'Road', 'Vegetation', 'Water']
imgs = ['VAZ1_201709290652_001_0037_L1A', 'VAZ1_201709290652_001_0053_L1A', 'VAZ1_201709291702_001_0023_L1A',
        'VAZ1_201711211359_001_0020_L1A', 'VAZ1_201711231340_002_0088_L1A', 'VAZ1_201711261310_001_0050_L1A',
        'VAZ1_201711261310_001_0086_L1A', 'VAZ1_201711290238_001_0061_L1A', 'VBZ1_201711231528_002_0023_L1A',
        'VBZ1_201711231528_002_0102_L1A', 'VBZ1_201711251154_001_0046_L1A', 'VBZ1_201711260636_001_0074_L1A']
img_path = '../data/Origin/'

col_step = 50
row_step = 50
size = 1024

def gen_data(obj_type):
    img_id = 0
    data_path = '../data/' + obj_type + '/'
    for img in imgs:
        mask_file = img_path + img + '_' + obj_type + '.tif'
        if not os.path.exists(mask_file):
            continue
        mask = tifffile.imread(mask_file)
        tif = tifffile.imread(img_path + img + '.tif')

        temp = tif[:, :, 0] + tif[:, :, 1] + tif[:, :, 2]
        tif_gray = np.zeros(temp.shape, dtype=np.uint8)
        tif_gray[np.where(temp == 0)] = 255

        width, height = tif.shape[:2]

        for j in range(0, width, col_step):
            if j % 500 == 0:
                print('rows:', j)
            for i in range(0, height, row_step):
                ttg = tif_gray[j:j + size, i:i + size]
                if np.sum(ttg == 0) >= 5:
                    continue
                tt = tif[j:j + size, i:i + size]
                tm = mask[j:j + size, i:i + size]
                tifffile.imsave(data_path + str(img_id) + '.tif', tt)
                tifffile.imsave(data_path + str(img_id) + '_mask.tif', tm)
                img_id += 1


if __name__ == '__main__':
    gen_data(obj_types[2])


'''
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
'''


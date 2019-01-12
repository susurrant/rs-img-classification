

import os, shutil
import tqdm
import numpy as np
import tifffile
import argparse


obj_types = ['Airport', 'Baresoil', 'Building', 'Farmland', 'Road', 'Vegetation', 'Water']
imgs = ['VAZ1_201709290652_001_0037_L1A', 'VAZ1_201709290652_001_0053_L1A', 'VAZ1_201709291702_001_0023_L1A',
        'VAZ1_201711211359_001_0020_L1A', 'VAZ1_201711231340_002_0088_L1A', 'VAZ1_201711261310_001_0050_L1A',
        'VAZ1_201711261310_001_0086_L1A', 'VAZ1_201711290238_001_0061_L1A', 'VBZ1_201711231528_002_0023_L1A',
        'VBZ1_201711231528_002_0102_L1A', 'VBZ1_201711251154_001_0046_L1A', 'VBZ1_201711260636_001_0074_L1A']
img_path = '../data/Origin/'


def gen_data(obj_type, col_step, row_step, size):
    img_id = 0
    data_path = '../data/' + obj_type + '/'
    shutil.rmtree(data_path)
    os.mkdir(data_path)

    for img in imgs:
        print(img)
        mask_file = img_path + img + '_' + obj_type + '.tif'
        if not os.path.exists(mask_file):
            print('  file does not exist.')
            continue
        mask = tifffile.imread(mask_file)
        width, height = mask.shape
        if width < size or height < size:
            continue

        tif = tifffile.imread(img_path + img + '.tif')
        temp = tif[:, :, 0] + tif[:, :, 1] + tif[:, :, 2]
        for j in tqdm.tqdm(range(0, width - size + 1, col_step)):
            for i in range(0, height - size + 1, row_step):
                ttg = temp[j:j + size, i:i + size]
                if j + size > width or i + size > height or np.sum(ttg == 0) >= 5:
                    continue
                tt = tif[j:j + size, i:i + size]
                tm = mask[j:j + size, i:i + size]
                tifffile.imsave(data_path + str(img_id) + '.tif', tt)
                tifffile.imsave(data_path + str(img_id) + '_mask.tif', tm)
                img_id += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '-type', help='object type', type=str)
    parser.add_argument('-c', '-col_step', help='column step', const=100, type=int) #default: 100
    parser.add_argument('-r', '-row_step', help='row step', type=int) #default: 100
    parser.add_argument('-s', '-size', help='image size', type=int) #default: 1024
    args = parser.parse_args()
    gen_data(obj_types[2], args.col_step, args.row_step, args.size)

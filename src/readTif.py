import cv2
import numpy as np
import tifffile

img = cv2.imread('../data/VBZ1_201711251154_001_0046_L1A.tif')
print(img.shape)

'''
v = set()

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        for k in range(3):
            v.add(img[i,j,k])
print(v)
'''
tem = img[:,:,0]
tem[np.where(tem==0)] = 255

#cv2.imwrite('gray.jpg', img)
cv2.namedWindow("Image")
cv2.imshow("Image", img[:,:])
cv2.waitKey (0)
cv2.destroyAllWindows()

'''
img = tifffile.imread('Raster_Airport.tif')
print(img.shape, np.max(img))
print(img[100])
'''
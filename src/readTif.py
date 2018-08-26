import cv2
import numpy as np
import tifffile
'''
img = cv2.imread('tif1.tif')
print(img.shape)

v = set()

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        for k in range(3):
            v.add(img[i,j,k])
print(v)

cv2.imwrite('gray.jpg', img)
cv2.namedWindow("Image")
cv2.imshow("Image", img[500:1500,500:1500])
cv2.waitKey (0)
cv2.destroyAllWindows()
'''

img = tifffile.imread('27566.tif')
print(img.shape, np.max(img))
print(img[0,0])
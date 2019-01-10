
import cv2
import numpy as np
import tifffile


img = tifffile.imread('VAZ1_201711261310_001_0050_L1A.tif')
a = img[:,:,0]+img[:,:,1]+img[:,:,2]
b = np.zeros(a.shape, dtype=np.uint8)
b[np.where(a == 0)] = 255

print(img.shape, np.max(img))

cv2.namedWindow("Image")
cv2.imshow("Image", cv2.resize(b, (img.shape[0]//10, img.shape[1]//10)))
cv2.waitKey (0)
cv2.destroyAllWindows()

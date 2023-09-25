import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('/media/yangxilab/DiskA1/zjy/datasets/horse2zebra/testB/n02391049_80.jpg',0)
edges = cv.Canny(img,400,450)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()

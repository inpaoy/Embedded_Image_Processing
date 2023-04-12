import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread(r"C:\Users\Inpao\Desktop\th (1).jfif")
b,g,r = cv2.split(img)
rgb_img = cv2.merge([r,g,b])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# noise removal
blurred = cv2.medianBlur(thresh, 5, 0)
#canny = cv2.Canny(blurred, 0, 255)

kernel = np.ones((3, 3), np.uint8)
#opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,kernel, iterations = 3)
closing = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel, iterations = 3)

# sure background area
sure_bg = cv2.dilate(closing, kernel, iterations=3)
sure_bg = np.uint8(sure_bg)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(sure_bg, cv2.DIST_L2, 3)

# Threshold
ret, sure_fg = cv2.threshold(dist_transform, 0.1*dist_transform.max(), 255, 0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# mark the region of unknown with zero
markers[unknown == 255] = 0

markers = cv2.watershed(img, markers)
img[markers == -1] = [255, 0, 0]

plt.subplot(421),plt.imshow(rgb_img)
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(422),plt.imshow(thresh, 'gray')
plt.title("Otsu's binary threshold"), plt.xticks([]), plt.yticks([])

plt.subplot(423),plt.imshow(closing, 'gray')
plt.title("morphologyEx:Closing:3x3"), plt.xticks([]), plt.yticks([])
plt.subplot(424),plt.imshow(sure_bg, 'gray')
plt.title("Dilation"), plt.xticks([]), plt.yticks([])

plt.subplot(425),plt.imshow(dist_transform, 'gray')
plt.title("Distance Transform"), plt.xticks([]), plt.yticks([])
plt.subplot(426),plt.imshow(sure_fg, 'gray')
plt.title("Thresholding"), plt.xticks([]), plt.yticks([])

plt.subplot(427),plt.imshow(unknown, 'gray')
plt.title("Unknown"), plt.xticks([]), plt.yticks([])

plt.subplot(428),plt.imshow(img, 'gray')
plt.title("Result from Watershed"), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()

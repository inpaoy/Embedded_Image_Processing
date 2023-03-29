import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r"C:\Users\Inpao\.vscode\Python\python1\road.jpg")

# ROI pick
showCrosshair = False
fromCenter = False
ROIs = cv2.selectROIs("Select ROIs", img, fromCenter, showCrosshair)
print(ROIs)

crop_number = 0

for rect in ROIs:
    x1 = rect[0]
    y1 = rect[1]
    x2 = rect[2]
    y2 = rect[3]

    img_crop = img[y1:y1+y2, x1:x1+x2]

    cv2.imshow("crop" +str(crop_number), img_crop)
    cv2.imwrite("crop" +str(crop_number)+ ".jpeg", img_crop)
    crop_number+=1

cv2.waitKey(0)
cv2.destroyAllWindows()

new_img1 = cv2.imread("crop0.jpeg")
new_img2 = cv2.imread("crop1.jpeg")
new_img3 = cv2.imread("crop2.jpeg")
new_img4 = cv2.imread("crop3.jpeg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray1 = cv2.cvtColor(new_img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(new_img2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(new_img3, cv2.COLOR_BGR2GRAY)
gray4 = cv2.cvtColor(new_img4, cv2.COLOR_BGR2GRAY)

# LBP
def LBP(gray):
    height, width = gray.shape[:2]
    dst = np.zeros((height, width), dtype = np.uint8)

    LBP_value = np.zeros((1, 8), dtype = np.uint8)
    neighbours = np.zeros((1, 8), dtype = np.uint8)
    for row in range(1, height - 1):
        for col in range(1, width - 1):
            center = gray[row, col]

            neighbours[0, 0] = gray[row - 1, col - 1]
            neighbours[0, 1] = gray[row - 1, col]
            neighbours[0, 2] = gray[row - 1, col + 1]
            neighbours[0, 3] = gray[row, col + 1]
            neighbours[0, 4] = gray[row + 1, col + 1]
            neighbours[0, 5] = gray[row + 1, col]
            neighbours[0, 6] = gray[row + 1, col - 1]
            neighbours[0, 7] = gray[row, col - 1]

            for i in range(8):
                if neighbours[0, i] > center:
                    LBP_value[0, i] = 1
                else:
                    LBP_value[0, i] = 0

            LBP = LBP_value[0, 0]*1 + LBP_value[0, 1]*2 + LBP_value[0, 2]*4 + LBP_value[0, 3]*8+ LBP_value[0, 4]*16+ LBP_value[0, 5]*32+ LBP_value[0, 6]*64+ LBP_value[0, 7]*128
            
            dst[row, col] = LBP
    return dst

#histogram
def lbp_histogram(gray):
    patterns = LBP(gray)
    hist, _ = np.histogram(patterns, bins=np.arange(2**8 + 1), density=True)
    return hist

img1_feats = lbp_histogram(gray1)
img2_feats = lbp_histogram(gray2)
img3_feats = lbp_histogram(gray3)
img4_feats = lbp_histogram(gray4)

hmax = max([img1_feats.max(), img2_feats.max(), img3_feats.max(), img4_feats.max()])
fig, ax = plt.subplots(2, 4)

ax[0, 0].imshow(new_img1)
ax[0, 0].axis('off')
ax[0, 0].set_title('img1')
ax[1, 0].plot(img1_feats)
ax[1, 0].set(xlabel = 'LBP level', ylabel = 'number of pixels')
ax[1, 0].set_ylim([0, hmax])

ax[0, 1].imshow(new_img2)
ax[0, 1].axis('off')
ax[0, 1].set_title('img2')
ax[1, 1].plot(img2_feats)
ax[1, 1].set(xlabel = 'LBP level')
ax[1, 1].set_ylim([0, hmax])
ax[1, 1].axes.yaxis.set_ticklabels([])

ax[0, 2].imshow(new_img3)
ax[0, 2].axis('off')
ax[0, 2].set_title('img3')
ax[1, 2].plot(img3_feats)
ax[1, 2].set(xlabel = 'LBP level')
ax[1, 1].set_ylim([0, hmax])
ax[1, 2].axes.yaxis.set_ticklabels([])

ax[0, 3].imshow(new_img4)
ax[0, 3].axis('off')
ax[0, 3].set_title('img4')
ax[1, 3].plot(img4_feats)
ax[1, 3].set(xlabel = 'LBP level')
ax[1, 1].set_ylim([0, hmax])
ax[1, 3].axes.yaxis.set_ticklabels([])
plt.show()


from scipy.spatial.distance import euclidean

e1 = euclidean(img1_feats, img2_feats)
e2 = euclidean(img1_feats, img3_feats)
e3 = euclidean(img1_feats, img4_feats)
e4 = euclidean(img2_feats, img3_feats)
e5 = euclidean(img2_feats, img4_feats)
e6 = euclidean(img3_feats, img4_feats)
print('\nimg1, img2: ', e1)
print('img1, img3: ', e2)
print('img1, img4: ', e3)
print('img2, img3: ', e4)
print('img2, img4: ', e5)
print('img3, img4: ', e6)

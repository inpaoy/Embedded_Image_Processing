import cv2
import time
import numpy as np

img = cv2.imread(r"C:\Users\Inpao\.vscode\python1\road.jpg")
img = cv2.resize(img, (600, 400))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray, 3)

def sobel(blur):
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.convertScaleAbs(sobely)
    sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    return sobelxy
    
sobel1 = sobel(blur)

runtime = 2000
kernel = np.ones((3, 3), np.uint8)
kernel2 = np.ones((5, 5), np.uint8)
kernel3 = np.ones((7, 7), np.uint8)
kernel4 = np.ones((9, 9), np.uint8)

#關閉AVX
cv2.setUseOptimized(False)
start = time.time()
for i in range(runtime):
    rst = cv2.morphologyEx(sobel1, cv2.MORPH_OPEN, kernel)
end = time.time()
time1 = end - start
print("AVX 關閉 3x3 kernel 所花的時間: ", time1) 

start = time.time()
for i in range(runtime):
    rst2 = cv2.morphologyEx(sobel1, cv2.MORPH_OPEN, kernel2)
end = time.time()
time2 = end - start
print("AVX 關閉 5x5 kernel 所花的時間: ", time2) 

start = time.time()
for i in range(runtime):
    rst3 = cv2.morphologyEx(sobel1, cv2.MORPH_OPEN, kernel3)
end = time.time()
time3 = end - start
print("AVX 關閉 7x7 kernel 所花的時間: ", time3) 

start = time.time()
for i in range(runtime):
    rst4 = cv2.morphologyEx(sobel1, cv2.MORPH_OPEN, kernel4)
end = time.time()
time4 = end - start
print("AVX 關閉 9x9 kernel 所花的時間: ", time4) 

#開啟AVX
cv2.setUseOptimized(True)
start = time.time()
for i in range(runtime):
    rst5 = cv2.morphologyEx(sobel1, cv2.MORPH_OPEN, kernel)
end = time.time()
time5 = end - start
print("AVX 開啟 3x3 kernel 所花的時間: ", time5)

start = time.time()
for i in range(runtime):
    rst6 = cv2.morphologyEx(sobel1, cv2.MORPH_OPEN, kernel2)
end = time.time()
time6 = end - start
print("AVX 開啟 5x5 kernel 所花的時間: ", time6) 

start = time.time()
for i in range(runtime):
    rst7 = cv2.morphologyEx(sobel1, cv2.MORPH_OPEN, kernel3)
end = time.time()
time7 = end - start
print("AVX 開啟 7x7 kernel 所花的時間: ", time7) 

start = time.time()
for i in range(runtime):
    rst8 = cv2.morphologyEx(sobel1, cv2.MORPH_OPEN, kernel4)
end = time.time()
time8 = end - start
print("AVX 開啟 5x5 kernel 所花的時間: ", time8) 

cv2.imshow('Original', img)
cv2.imshow('ksize3', rst)
cv2.imshow('ksize5', rst2)
cv2.imshow('ksize7', rst3)
cv2.imshow('ksize9', rst4)
cv2.waitKey()
cv2.destroyAllWindows()

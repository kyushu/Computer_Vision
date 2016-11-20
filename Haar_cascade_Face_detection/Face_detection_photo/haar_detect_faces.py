
# -*- coding: utf-8 -*-

# usage: python haar_detect_faces.py -f ../cascades/haarcascade_frontalface_default.xml -i images/adrian.png

'''
Viola-Jones algorithm  (also known as Haar cascades)

容易 false-positive detection 跟 沒偵測到臉
常常要針對不同的 image 調參數 (it can be a real pain)

'''

from __future__ import print_function
import argparse
import cv2

ap = argparse.ArgumentParser()
# 使用 OpenCV pre-train classifier 的 XML 檔案
ap.add_argument("-f", "--face", required=True, help="Path to where the face cascade resides")
ap.add_argument("-i", "--image", required=True, help="Path to where the image file resides")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

'''
cv2.CascadeClassifier.detectMultiScale(image[, scaleFactor[, minNeighbors[, flags[, minSize[, maxSize]]]]]) → objects

image : source
scaleFactor     : How much the image size is reduced at each image scale. This value is used to
                    create the scale pyramid. In order to detect faces at multiple scales in the
                    image (some faces may be closer to the foreground, and thus be larger, other
                    faces may be smaller and in the background, thus the usage of varying scales).
                    A value of 1.05 indicates that we are reducing the size of the image by 5%
                    at each level in the pyramid

minNeighbors    : Parameter specifying how many neighbors each candidate rectangle should
                    have to retain it

flags           : Parameter with the same meaning for an old cascade as in the function
                    cvHaarDetectObjects. It is not used for a new cascade

minSize         : Minimum possible object size. Objects smaller than that are ignored

maxSize         : Maximum possible object size. Objects larger than that are ignored



影響結果最明顯的是
1. scaleFactor
    大部份情況下很容易發生 scaleFactor 對這個 image 偵測正常，但對另一個 image 偵測會 false-positive
    或是 miss detection

2. minNeighbors
    有些情況下會是 minNeighbors 導致的

基本上從 scaleFactor 修改，再改 minNeighbors

'''
detector = cv2.CascadeClassifier(args["face"])
# faceRects = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30,30), flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
faceRects = detector.detectMultiScale(gray, scaleFactor=1.06, minNeighbors=5, minSize=(30,30), flags = cv2.cv.CV_HAAR_SCALE_IMAGE)


print("i found %d face(s)" % (len(faceRects)))


for (x, y, w, h) in faceRects:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces", image)
cv2.waitKey(0)


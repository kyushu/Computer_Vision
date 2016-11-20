# -*- coding:utf8 -*-

# Usage: python lbp_faces.py --dataset ./datasets/caltech_faces/

from __future__ import print_function
from mtImageSearch.face_recognition import load_caltech_faces
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to CALTECH faces dataset")
ap.add_argument("-s", "--sample-size", type=int, default=10, help="# of example samples")
args = vars(ap.parse_args())


#
# 1.
#
print("[INFO] loading CALTECH Faces dataset...")
# min_face : if min_face = 21, we must have at least 21 sample of their face
# test_size: if test_size = 0.25, we are using 75% data for training and 25% data for testing
(training, testing, names) = load_caltech_faces(args["dataset"], min_faces=21, test_size=0.25)

# encode the string labels as unique integer labels, transforming them from strings into integers
# 因為 OpenCV 內建的 LBP face recognizer 只吃 integer label
le = LabelEncoder()
le.fit_transform(training.target)


#
# 2. Training the Local Binary Pattern face recognizer
#
print("[INFO] training face recognizer...")
'''
grid_x, grid_y controls the number of MxN cell in the feac recoginition algorithm
LBP Face recognition 的論文是建議 7x7, 不過這裡用 8x8 是因為切分更細可以得到更高的準確度
more granularity resulting in more accuracy
不過原本只算 7x7=49 個 LBP histogram, 現在要算 8x8=64 個 LBP histogram
所以會導致
1. longer feature extraction/comparison times
2. more memory consumption to store the feature vectors
'''
recognizer = cv2.createLBPHFaceRecognizer(radius=2, neighbors=16, grid_x=3, grid_y=3)

'''
FaceRecognizer.train(InputArrayOfArrays src, InputArray labels)
src     : array of image
labels  : array of int
'''
recognizer.train(training.data, le.transform(training.target))


#
# 3. Gathering predictions
#
print("[INFO] gathering predictions...")
predictions = []
confidence = []

# loop over the test data
for i in xrange(0, len(testing.data)):
    '''
    FaceRecognizer.predict(src) return (label, confidence)
    confidence : the closest distance between test data and train data
    '''
    (prediction, conf) = recognizer.predict(testing.data[i])
    predictions.append(prediction)
    confidence.append(conf)

# show the classification report
print(classification_report(le.transform(testing.target), predictions, target_names=names))


#
# 4. display prediction reault
#
for i in np.random.randint(0, high=len(testing.data), size=(args["sample_size"],)):
    print("[INFO] Prediction: {}, Actual: {}, Confidence: {:.2f}".format(le.inverse_transform(predictions[i]), testing.target[i], confidence[i]))

    face = testing.data[i]
    face = imutils.resize(face, width=face.shape[1] * 2, inter=cv2.INTER_CUBIC)
    cv2.imshow("Face", face)
    cv2.waitKey(0)

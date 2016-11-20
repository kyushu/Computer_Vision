# -*- coding:utf-8 -*-

'''
usage:
    python train_simple.py --fonts input/example_fonts --char-classifier output/simple_char.cpickle --digit-classifier output/simple_digit.cpickle

'''

from __future__ import print_function
from mtImageSearch.descriptors import BlockBinaryPixelSum
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from imutils import paths
import argparse
import cPickle
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--fonts", required=True, help="path to the fonts dataset")
ap.add_argument("-c", "--char-classifier", required=True, help="path to the output character classifier")
ap.add_argument("-d", "--digit-classifier", required=True, help="path to the output digit classifier")
args = vars(ap.parse_args())


alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"

# initialize the data and labels for the alphabet and digits
alphabetData = []
digitsData = []
alphabetLabels = []
digitsLabels = []


print("[INFO] describing font examples...")
blockSizes = ((5, 5), (5, 10), (10, 5), (10, 10))
# initialize BlockBinaryPixelSum with parameters
desc = BlockBinaryPixelSum(targetSize=(30, 15), blockSizes=blockSizes)

# loop over the font from its path
for fontPath in paths.list_images(args["fonts"]):
    # load the font image
    font = cv2.imread(fontPath)
    # convert it to Grayscale
    font = cv2.cvtColor(font, cv2.COLOR_BGR2GRAY)
    # threshold font, > 128 = 255
    #                 < 128 = 0
    thresh = cv2.threshold(font, 128, 255, cv2.THRESH_BINARY_INV)[1]

    # get contours
    (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print("conts type:{}".format(type(cnts)))
    # sort contours by it coordinate (from left-top to right-bottom)
    cnts = sorted(cnts, key=lambda c:(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[1]))
    # print("cnts: {}".format(cnts))
    # loop over th contours
    for (i, c) in enumerate(cnts):
        # get bounding box for the contour
        (x, y, w, h) = cv2.boundingRect(c)
        # get ROI
        roi = thresh[y:y + h, x:x + w]
        # cv2.imshow("roi", roi)
        # cv2.waitKey(0)
        # describe ROI
        features = desc.describe(roi)
        # print("i={}".format(i))
        if i < 26:
            # alphabet
            alphabetData.append(features)
            alphabetLabels.append(alphabet[i])
            # print("features: {}: {}".format(alphabet[i], features))
            # cv2.imshow("{}".format(alphabet[i]), roi)
            # cv2.waitKey(0)
        else:
            # digits
            digitsData.append(features)
            digitsLabels.append(alphabet[i])
            # print("features: {}: {}".format(alphabet[i], features))
            # cv2.imshow("{}".format(alphabet[i]), roi)
            # cv2.waitKey(0)
        # cv2.destroyAllWindows()


# 1. train the character classifier
print("[INFO] fitting character model...")
charModel = LinearSVC(C=1.0, random_state=42)
charModel.fit(alphabetData, alphabetLabels)
score = charModel.score(alphabetData, alphabetLabels)
print("char accuracy: {}".format(score))

# 2. train the digit classifier
print("[INFO] fitting digit model...")
digitModel = LinearSVC(C=1.0, random_state=42)
digitModel.fit(digitsData, digitsLabels)
score = digitModel.score(digitsData, digitsLabels)
print("digit accuracy: {}".format(score))

# 3. Store the character classifier(trained model) to file
print("[INFO] store character model...")
f = open(args["char_classifier"], "w")
f.write(cPickle.dumps(charModel))
f.close()

# 4. store the character classifier(trained model) to file
print("[INFO] store digit model...")
f = open(args["digit_classifier"], "w")
f.write(cPickle.dumps(digitModel))
f.close()

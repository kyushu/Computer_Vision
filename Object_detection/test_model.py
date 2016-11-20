# -*- coding: utf-8 -*-


'''
usage:
    python test_model.py --conf conf/cars_side.json --image ../datasets/caltech101/101_ObjectCategories/car_side/image_0004.jpg
    python test_model.py --conf conf/cars_side.json --image ../datasets/caltech101/101_ObjectCategories/car_side/image_0016.jpg
'''



# import the necessary packages
from mtimagesearch.object_detection import non_max_suppression
from mtimagesearch.object_detection import ObjectDetector
from mtimagesearch.descriptors import HOG
from mtimagesearch.utils import Conf
import numpy as np
import imutils
import argparse
import cPickle
import cv2
import datetime

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to the configuration file")
ap.add_argument("-i", "--image", required=True, help="path to the image to be classified")
args = vars(ap.parse_args())

# load the configuration file
conf = Conf(args["conf"])

# load the classifier, then initialize the Histogram of Oriented Gradients descriptor
# and the object detector
model = cPickle.loads(open(conf["classifier_path"]).read())
hog = HOG(orientations=conf["orientations"], pixelsPerCell=tuple(conf["pixels_per_cell"]),
    cellsPerBlock=tuple(conf["cells_per_block"]), normalize=conf["normalize"])
od = ObjectDetector(model, hog)



# load the image and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=min(260, image.shape[1]))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print("[INFO]:{} Starting Detect".format(datetime.datetime.now()))
# detect objects in the image and apply non-maxima suppression to the bounding boxes
(boxes, probs) = od.detect(gray, conf["window_dim"], winStep=conf["window_step"],
    pyramidScale=conf["pyramid_scale"], minProb=conf["min_probability"])

print("[INFO]:{} Starting Non Max Suppression".format(datetime.datetime.now()))
pick = non_max_suppression(np.array(boxes), probs, conf["overlap_thresh"])
orig = image.copy()

print("Draw Bounding box without NMS")
# loop over the original bounding boxes and draw them
for (startX, startY, endX, endY) in boxes:
    cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 0, 255), 2)

print("Draw Bounding box with NMS")
# loop over the allowed bounding boxes and draw them
for (startX, startY, endX, endY) in pick:
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

# show the output images
cv2.imshow("Original", orig)
cv2.imshow("Image", image)
cv2.waitKey(0)

# -*- coding: utf-8 -*-

# usage: python test_model_no_nms.py --conf conf/cars_side.json --image ../datasets/caltech101/101_ObjectCategories/car_side/image_0004.jpg

from mtimagesearch.object_detection import ObjectDetector
from mtimagesearch.descriptors import HOG
from mtimagesearch.utils import Conf
import imutils
import argparse
import cPickle
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="Path to the configuration file")
ap.add_argument("-i", "--image", required=True, help="Path to the image to be classified")
args = vars(ap.parse_args())

conf = Conf(args["conf"])

# 1. Load the trained Linear SVM model
model = cPickle.loads(open(conf["classifier_path"]).read())
# 2. Initialize the HOG descriptor
hog = HOG(orientations=conf["orientations"], pixelsPerCell=tuple(conf["pixels_per_cell"]),
    cellsPerBlock=tuple(conf["cells_per_block"]), normalize=conf["normalize"])
# 3. Initialize Object Detector
od = ObjectDetector(model, hog)

# 4. Load the image
image = cv2.imread(args["image"])
# 4-1. resize image to width = 260
image = imutils.resize(image, width=min(260, image.shape[1]))
# 4-2. convert color space to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 5. Detect the Object (detect what object is dependent on our configuration file)
(boxes, probs) = od.detect(gray, conf["window_dim"], winStep=conf["window_step"],
    pyramidScale=conf["pyramid_scale"], minProb=conf["min_probability"])

# Draw the bounding box of what we detect
for (startX, startY, endX, endY) in boxes:
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)

cv2.imshow("Image", image)
cv2.waitKey(0)


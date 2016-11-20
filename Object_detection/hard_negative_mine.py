# -*- coding:utf-8 -*-

'''
usage:
    python hard_negative_mine.py --conf conf/cars_side.json
'''

from __future__ import print_function
from mtimagesearch.object_detection import ObjectDetector
from mtimagesearch.descriptors import HOG
from mtimagesearch.utils import dataset
from mtimagesearch.utils import Conf
from imutils import paths
import numpy as np
import progressbar
import argparse
import cPickle
import random
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to the configuration file")
args = vars(ap.parse_args())

conf = Conf(args["conf"])
data = []

# load pre-trained classifier
model = cPickle.loads(open(conf["classifier_path"]).read())
# initialize the Histogram of Oriented Gradients descriptor
hog = HOG(orientations=conf["orientations"], pixelsPerCell = tuple(conf["pixels_per_cell"]), cellsPerBlock=tuple(conf["cells_per_block"]), normalize=conf["normalize"])

# initialize the Object-Detector with classifier and descriptor
'''
這裡用 已訓練好偵測我們要的物件 的 classifier 來偵測 negative images
negative image 表示 image 裡沒有我們要的物件，所以若用這個 classifier 有偵測到物件
就表示是 false-positive

舉例來說， classifier 已訓練為偵測 car_side 的物件
而 negative image set 裡是一堆沒有 car_side 的 風景照
所以用這個 classifier 來偵測這些風景照所偵測到的 object 就是 false-positive
'''
od = ObjectDetector(model, hog)

# grab the set of negative image and random sample them
dstPaths = list(paths.list_images(conf["image_distractions"]))
dstPaths = random.sample(dstPaths, conf["hn_num_distraction_images"])

# set up the Progress bar
widgets = ["Mining: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(dstPaths), widgets=widgets).start()

'''
這裡用
'''
# loop over the negative image from path
for (i, imagePath) in enumerate(dstPaths):
    # load image from path
    image = cv2.imread(imagePath)
    # convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect object in the image
    (boxes, probs) = od.detect(gray, conf["window_dim"], winStep=conf["hn_window_step"], pyramidScale=conf["hn_pyramid_scale"], minProb=conf["hn_min_probability"])

    # loop over the bounding box
    for (prob, (startX, startY, endX, endY)) in zip(probs, boxes):

        # extract ROI from the image -> resize it to a Known canonical size
        roi = cv2.resize(gray[startY:endY, startX:endX], tuple(conf["window_dim"]), interpolation=cv2.INTER_AREA)
        # extract the HOG features from the resized ROI
        features = hog.describe(roi)
        # update the data
        data.append(np.hstack([[prob], features]))

    pbar.update(i)



# sort the data points by confidence
pbar.finish()
print("[INFO] sorting by probability...")
data = np.array(data)
data = data[data[:, 0].argsort()[::-1]]

# dump the dataset to file
print("[INFO] dumping hard negatives to file...")
dataset.dump_dataset(data[:, 1:], [-1] * len(data), conf["features_path"], "hard_negatives",
    writeMethod="a")



# -*- coding:utf-8 -*-

'''
usage:
    python index_features.py --dataset output/data/training --features-db output/training_features.hdf5
'''

# import the necessary packages
from __future__ import print_function
from mtImageSearch.descriptors import DetectAndDescribe
from mtImageSearch.descriptors import RootSIFT
from mtImageSearch.indexer import FeatureIndexer
from imutils import paths
import argparse
import imutils
import random
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to the directory that contains the images to be indexed")
ap.add_argument("-f", "--features-db", required=True, help="Path to where the features database will be stored")
ap.add_argument("-a", "--approx-images", type=int, default=250, help="Approximate # of images in the dataset")
ap.add_argument("-b", "--max-buffer-size", type=int, default=50000, help="Maximum buffer size for # of features to be stored in memory")
args = vars(ap.parse_args())

# initialize the keypoint detector
detector = cv2.FeatureDetector_create("GFTT")
# initialize the featrue descriptor
descriptor = RootSIFT()
# initialize the object include keypoint detector and feature descriptor
dad = DetectAndDescribe(detector, descriptor)

# initialize the feature indexer
fi = FeatureIndexer(args["features_db"], estNumImages=args["approx_images"],
    maxBufferSize=args["max_buffer_size"], verbose=True)

#
imagePaths = list(paths.list_images(args["dataset"]))
random.shuffle(imagePaths)


for (i, imagePath) in enumerate(imagePaths):
    if i > 0 and i % 10 == 0:
        fi._debug("processed {} images".format(i), msgType="[PROGRESS]")

    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=320)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    (kps, descs) = dad.describe(image)

    if kps is None or descs is None:
        continue


    (label, filename) = imagePath.split("/")[-2:]
    k = "{}:{}".format(label, filename)
    # 這裡多存了 image.shape,
    fi.add(k, image.shape, kps, descs)

fi.finish()


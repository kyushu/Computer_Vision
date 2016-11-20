# -*- coding:utf-8 -*-

# usage: python extract_feature --conf ./conf/cars_side.json

'''
Purpose:
    Extract and label features from Positive image dataset and Negative image dataset
    then store into HDF5 file
'''


from __future__ import print_function
from sklearn.feature_extraction.image import extract_patches_2d
from mtimagesearch.object_detection import helpers
from mtimagesearch.descriptors import HOG
from mtimagesearch.utils import dataset
from mtimagesearch.utils import Conf
from imutils import paths
from scipy import io
import numpy as np
import progressbar
import argparse
import random
import cv2


# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="Path to the configuration file")
args = vars(ap.parse_args())

# parameters of configure
conf = Conf(args["conf"])

# initialize HOG detector with corresponding parameters
hog = HOG(orientations=conf["orientations"], pixelsPerCell=tuple(conf["pixels_per_cell"]),
    cellsPerBlock=tuple(conf["cells_per_block"]), normalize=conf["normalize"])


# data for store features
data = []
# labels for Positive or Negative label
labels = []

#####################################################################################################
# Get image paths for Positive Image
#
trainPaths = list(paths.list_images(conf["image_dataset"]))
# Ramdomly sample 50% images to be training data
trainPaths = random.sample(trainPaths, int(len(trainPaths) * conf["percent_gt_images"]))
print("[INFO] describing training ROIs...")

# setUp widget and pbar
widgets = ["Extracting: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(trainPaths), widgets=widgets).start()

# loop features of Positive image
for (i, trainPath) in enumerate(trainPaths):
    # load image from train path
    image = cv2.imread(trainPath)
    # convert image from RGB to Grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 檔案格式 ＝ image_0001.jpg
    # 找到 0001.jpg 後再將 .jpg 去掉
    imageID = trainPath[trainPath.rfind("_") + 1:].replace(".jpg", "")

    p = "{}/annotation_{}.mat".format(conf["image_annotations"], imageID)
    '''
    the content of annotation in caltech101/Annotations
    '__globals__': [],
    '__header__': 'MATLAB 5.0 MAT-file, Platform: PCWIN, Created on: Tue Dec 14 11:03:29 2004',
    '__version__': '1.0',
    'box_coord': array([[ 30, 137,  49, 349]], dtype=uint16),
    'obj_contour': a numpy.ndarray contain contour
    '''
    bb = io.loadmat(p)["box_coord"][0]
    # get ROI from image by "window_dim" size
    roi = helpers.crop_ct101_bb(image, bb, padding=conf["offset"], dstSize=tuple(conf["window_dim"]))

    rois = (roi, cv2.flip(roi, 1)) if conf["use_flip"] else (roi,)
    for roi in rois:
        features = hog.describe(roi)
        data.append(features)
        # label as 1 (Positive)
        labels.append(1)

    pbar.update(i)
pbar.finish()

#####################################################################################################
# Get distraction image paths for Negative image
#
dstPaths = list(paths.list_images(conf["image_distractions"]))
# reset pbar
pbar = progressbar.ProgressBar(maxval=conf["num_distraction_images"], widgets=widgets).start()
print("[INFO] describing distraction ROIs...")

# loop features of Negative image
for i in np.arange(0, conf["num_distraction_images"]):
    image = cv2.imread(random.choice(dstPaths))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    '''
    sklearn.feature_extraction.image.extract_patches_2d(image, patch_size, max_patches=None, random_state=None)
    image : array, shape = (image_height, image_width) or
            (image_height, image_width, n_channels) The original image data.
            For color images, the last dimension specifies the channel: a RGB image would have n_channels=3.
    patch_size  : tuple of ints (patch_height, patch_width) the dimensions of one patch
    max_patches : integer or float, optional default is None
                  The maximum number of patches to extract.
                  If max_patches is a float between 0 and 1, it is taken to be a proportion of the total number of patches.
    random_state : int or RandomState Pseudo number generator state used for random sampling to use
                   if max_patches is not None.

    return : patches : array, shape = (n_patches, patch_height, patch_width) or
                       (n_patches, patch_height, patch_width, n_channels) The collection of
                       patches extracted from the image, where n_patches is either max_patches or
                       the total number of patches that can be extracted.

    extract_patches_2d 將 image 以 patch_size 的 sliding_window 由左至右取出
    '''
    patches = extract_patches_2d(image, tuple(conf["window_dim"]), max_patches=conf["num_distractions_per_image"])
    # 取得 Negative image patch 的 HOG Features
    for patch in patches:
        features = hog.describe(patch)
        data.append(features)
        # label as -1 (Negative)
        labels.append(-1)

    pbar.update()
pbar.finish()


# dump the dataset to file (data and labels)
print("[INFO] dumping features and labels to file...")
dataset.dump_dataset(data, labels, conf["features_path"], "features")

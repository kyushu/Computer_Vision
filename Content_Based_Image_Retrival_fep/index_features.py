
# -*- coding: utf-8 -*-

'''
usage: python index_features.py --dataset ../ukbench_samples/ --features-db output/features.hdf5

1. Extracting keypoints and features from the image dataset.
2. Clustering the extracted features using k-means to form a codebook.
3. Constructing a bag-of-visual-words (BOVW) representation for each image by quantizing the feature vectors associated with each image into a histogram using the codebook from Step 2.
4. Accepting a query image from the user, constructing the BOVW representation for the query, and performing the actual search
'''


from __future__ import print_function
from mtImageSearch.descriptors import DetectAndDescribe
from mtImageSearch.descriptors import RootSIFT
from mtImageSearch.indexer import FeatureIndexer
from imutils import paths
import argparse
import imutils
import cv2

'''
dataset         : 選擇要做 feature extraction 的 image 資料
features-db     : feature extraction 的資料要儲存的路徑，這裡使用 HDF5 database 來儲存
approx-images   : dataset 裡 image 的個數，因為 HDF5 database 需給定大小來決定 database 的 size
                    若之後超出這個 size 則會花更多的時間來做
max-buffer-size : 直接寫資料(feature vector)到 HDF5 是很慢的，不過將資料寫到 memory buffer 再寫到 HDF5
                    則快很多，所以我們需要定義 memory buffer 大小，當 buffer 滿了就寫到 HDF5 去
'''

#
# Step 1. Define Parameter
#
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to the directory that contains the images to be indexed")
ap.add_argument("-f", "--features-db", required=True, help="Path to where the features database will be stored")
ap.add_argument("-a", "--approx-images", type=int, default=500, help="Approximate # of images in the dataset")
ap.add_argument("-b", "--max-buffer-size", type=int, default=50000, help="Maximum buffer size for # of features to e cstored in memory")
args = vars(ap.parse_args())

#
# Step 2. Initialize All we need objects
#
# initialize the keypoint detector
detector = cv2.FeatureDetector_create("SURF") # Fast Hessian method
# initialize the local invariant descriptor
descriptor = RootSIFT()
# initialize the descriptor pipeline
dad = DetectAndDescribe(detector, descriptor)
# initialize the feature indexer
fi = FeatureIndexer(args["features_db"], estNumImages=args["approx_images"], maxBufferSize=args["max_buffer_size"], verbose=True)

# loop over the images in the dataset
for (i, imagePath) in enumerate(paths.list_images(args["dataset"])):
    # check to see if progress should be displayed
    if i > 0 and i % 10 == 0:
        fi._debug("processed {} images".format(i), msgType="[PROGRESS")

    # extract the image filename (i.e. the unique image ID)
    filename = imagePath[imagePath.rfind("/") + 1:]
    # load image
    image = cv2.imread(imagePath)
    # resize image
    image = imutils.resize(image, width=320)
    # convert iamge to gray-scale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # describe the image
    (kps, descs) = dad.describe(image)
    print("kps: {}".format(kps))
    # if either the keypoints or descriptors are None, the ignore the image
    if kps is None or descs is None:
        continue

    # index the features
    fi.add(filename, kps, descs)

# finish the indexing process
fi.finish()






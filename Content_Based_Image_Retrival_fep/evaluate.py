# -*- coding:utf-8 -*-

'''
usage:
python evaluate.py --dataset ../ukbench_samples --features-db output/features.hdf5 --bovw-db output/bovw.hdf5 --codebook output/vocab.cpickle --relevant ../ukbench_samples/relevant.json --idf output/idf.cpickle

'''

# import the necessary packages
from __future__ import print_function
from mtImageSearch.descriptors import DetectAndDescribe
from mtImageSearch.descriptors import RootSIFT
from mtImageSearch.ir import BagOfVisualWords
from mtImageSearch.ir import Searcher
from mtImageSearch.ir import dists
from scipy.spatial import distance
from redis import Redis
import numpy as np
import progressbar
import argparse
import cPickle
import imutils
import json
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to the directory of indexed images")
ap.add_argument("-f", "--features-db", required=True, help="Path to the features database")
ap.add_argument("-b", "--bovw-db", required=True, help="Path to the bag-of-visual-words database")
ap.add_argument("-c", "--codebook", required=True, help="Path to the codebook")
ap.add_argument("-i", "--idf", type=str, help="Path to inverted document frequencies array")
ap.add_argument("-r", "--relevant", required=True, help="Path to relevant dictionary")
args = vars(ap.parse_args())


# initialize the keypoint detector, local invariant descriptor, descriptor pipeline,
# distance metric, and inverted document frequency array
detector = cv2.FeatureDetector_create("SURF")
descriptor = RootSIFT()
dad = DetectAndDescribe(detector, descriptor)
distanceMetric = dists.chi2_distance
idf = None

# if the path to the inverted document frequency array was supplied, then load the
# idf array and update the distance metric
if args["idf"] is not None:
    idf = cPickle.loads(open(args["idf"]).read())
    distanceMetric = distance.cosine

# load the codebook vocabulary and initialize the bag-of-visual-words transformer
vocab = cPickle.loads(open(args["codebook"]).read())
bovw = BagOfVisualWords(vocab)

# connect to redis and initialize the searcher
redisDB = Redis(host="localhost", port=6379, db=0)
searcher = Searcher(redisDB, args["bovw_db"], args["features_db"], idf=idf,
    distanceMetric=distanceMetric)


# load the relevant queries dictionary
relevant = json.loads(open(args["relevant"]).read())
queryIDs = relevant.keys()

# initialize the accuracies list and the timings list
accuracies = []
timings = []

# initialize the progress bar
widgets = ["Evaluating: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(queryIDs), widgets=widgets).start()


# loop over the images
for (i, queryID) in enumerate(sorted(queryIDs)):

    # look up the relevant results for the query image
    # queryRlevant 是一個 image ID array
    queryRelevant = relevant[queryID]

    # load the query image and process it
    p = "{}/{}".format(args["dataset"], queryID)
    queryImage = cv2.imread(p)
    queryImage = imutils.resize(queryImage, width=320)
    queryImage = cv2.cvtColor(queryImage, cv2.COLOR_BGR2GRAY)

    # extract features from the query image and construct a bag-of-visual-words
    # from it
    (_, descs) = dad.describe(queryImage)
    hist = bovw.describe(descs).tocoo()

    # perform the search and compute the total number of relevant images in the
    # top-4 results
    sr = searcher.search(hist, numResults=4)
    results = set([r[1] for r in sr.results])
    # intersection : 取得兩個 set 的交集
    inter = results.intersection(queryRelevant)

    # update the evaluation lists
    # 交集的個數就是分數
    accuracies.append(len(inter))
    timings.append(sr.search_time)
    pbar.update(i)


# release any pointers allocated by the searcher
searcher.finish()
pbar.finish()

# show evaluation information to the user
accuracies = np.array(accuracies)
timings = np.array(timings)
print("[INFO] ACCURACY: u={:.2f}, o={:.2f}".format(accuracies.mean(), accuracies.std()))
print("[INFO] TIMINGS: u={:.2f}, o={:.2f}".format(timings.mean(), timings.std()))

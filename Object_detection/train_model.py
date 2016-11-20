# -*- coding:utf-8 -*-


'''
usage:
    python train_model.py --conf ./conf/cars_side.json
    python train_model.py --conf ./conf/cars_side.json --hard-negatives 1

Purpose:
    read features from hdf5 file which has extracted by "extract_feature.py"
    train SVM by this features, then dump model to file(cPickle)
'''

from __future__ import print_function
from mtimagesearch.utils import dataset
from mtimagesearch.utils import Conf
from sklearn.svm import SVC
import numpy as np
import argparse
import cPickle

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="Path to the configuration file")

# hard-negatives is -1 if we don't have any negative features
ap.add_argument("-n", "--hard-negatives", type=int, default=-1,
    help="flag indicating whether or not hard negatives should be used")
args = vars(ap.parse_args())

# 1. Load Configuration parameters
print("[INFO] loading dataset...")
conf = Conf(args["conf"])
(data, labels) = dataset.load_dataset(conf["features_path"], "features")

# 2. Load hard negatives
if args["hard_negatives"] > 0:
    print("[INFO] Loading hard negatives...")
    (hardData, hardLabels) = dataset.load_dataset(conf["features_path"], "hard_negatives")
    data = np.vstack([data, hardData])
    labels = np.hstack([labels, hardLabels])


# 3. Train the classifier
print("[INFO] Training classifier...")
model = SVC(kernel="linear", C=conf["C"], probability=True, random_state=42)
model.fit(data, labels)

# 4. Dump the classifier to file
print("[INFO] Dumping classifier...")
f = open(conf["classifier_path"], "w")
f.write(cPickle.dumps(model))
f.close()


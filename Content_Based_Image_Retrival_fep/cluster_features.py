# -*- coding:utf-8 -*-

# usage: python cluster_features.py --features-db output/features.hdf5 --codebook output/vocab.cpickle --clusters 1536 --percentage0.25


'''
如何決定 codebook 的 size
    1. Deciding on a set of cluster sizes K to evaluate.
    2. Clustering the feature vectors using each k in K.
    3. Evaluating the performance of the codebook according to some metric (ex. raw accuracy, precision, recall, f-measure, etc.)

    一般而言，你會想用 smaller, dense codebook 來做 machine-learning，
    但是當你想在 10,000 個 motocycle 的 dataset 裡找出特定的 motocycle 時
    這時 larger, sparse codebook 就會很有用了
'''


from __future__ import print_function
from mtImageSearch.ir import Vocabulary
import argparse
import cPickle

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--features-db", required=True, 
    help="Path to where the features database will be stored")
ap.add_argument("-c", "--codebook", required=True, 
    help="Path to the output codebook")
ap.add_argument("-k", "--clusters", type=int, default=64, 
    help="# of clusters to generate")
ap.add_argument("-p", "--percentage", type=float, default=0.25,
    help="Percentage of total features to use when clustering")

args = vars(ap.parse_args())

# Create the visual words vocabulary
voc = Vocabulary(args["features_db"])
vocab = voc.fit(args["clusters"], args["percentage"])

# Dump the clusters to file (cluster's centroid)
print("[INFO] storing cluster centers...")

'''
open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True)
mode :
    r = read (default)
    w = write
    a = append
    b = binary
    t = text (default)
    + = 更新磁碟檔案
    U = 通用新行模式
'''
f = open(args["codebook"], "w")

f.write(cPickle.dumps(vocab))
f.close()


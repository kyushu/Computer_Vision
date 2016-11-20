# -*- coding:utf-8 -*-

'''
usage:
python cluster_features.py --features-db output/training_features.hdf5 --codebook output/vocab.cpickle --clusters 512 --percentage 0.25
'''


from __future__ import print_function
from mtImageSearch.ir import Vocabulary
import argparse
import cPickle

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--features-db", required=True, help="Path to the features database")
ap.add_argument("-c", "--codebook", required=True, help="Path to the output codebook")
ap.add_argument("-k", "--clusters", type=int, default=64, help="# of clusters to generate")
ap.add_argument("-p", "--percentage", type=float, default=0.25, help="Percentage of total features to user when clustering")
args = vars(ap.parse_args())

voc = Vocabulary(args["features_db"])
vocab = voc.fit(args["clusters"], args["percentage"])

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
f= open(args["codebook"], "w")

f.write(cPickle.dumps(vocab))
f.close()






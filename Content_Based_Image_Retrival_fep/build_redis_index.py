# -*- coding:utf-8 -*-

'''
usage: python build_redis_index.py --bovw-db output/bovw.hdf5
'''

from __future__ import print_function
from mtImageSearch.db import RedisQueue
from redis import Redis
import argparse
import h5py

ap = argparse.ArgumentParser()
ap.add_argument("-b", "--bovw-db", required=True, help="Path to where the bag-of-visual-words database")
args = vars(ap.parse_args())

# Connect to Redis Server
redisDB = Redis(host="localhost", port=6379, db=0)
rq = RedisQueue(redisDB)

# Load the bag-of-visual-words database from HDF5 file
bovwDB = h5py.File(args["bovw_db"], mode="r")

# Loop over the entries in the bag-of-visual-words database
for (i, hist) in enumerate(bovwDB["bovw"]):
    if i > 0 and i % 10 == 0:
        # 每 10 筆列一次資訊
        print("[PROGRESS] processed {} enries".format(i))
    # 將 BOVW 存入對應的 imageId list
    rq.add(i, hist)

bovwDB.close()
rq.finish()

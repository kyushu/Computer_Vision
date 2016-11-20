# -*- coding:utf-8 -*-

from __future__ import print_function
'''
MiniBatchKMeans
    MiniBatchKMeans 是將 dataset 切成較小的 segment，對每個 segment 個別做 clustering
    最後再將每個 segment clustering 的結果合併成最後的結果

    MiniBatchKMeans 並沒有比 K-Means 來得 accurate，基本上應該是比較差的
    但 MiniBatchKMeans 比 K-Means 快很多
'''
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import datetime
import h5py

class Vocabulary:
    def __init__(self, dbPath, verbose=True):
        self.dbPath = dbPath
        self.verbose = verbose

    '''
    numClusters : 有多少個 cluster
    samplePercent : 一次要從 feature database 裡取多少 sample 出來做 mini-batch k-means
    '''
    def fit(self, numClusters, samplePercent, randomState=None):
        # 載入 feature database
        db = h5py.File(self.dbPath)
        totalFeatures = db["features"].shape[0]

        # 根據 samplePercent 決定要從 database 裡取出的 sample 數
        # 再用 np.random.choice 隨機從 database 裡取出 sample
        sampleSize = int(np.ceil(samplePercent * totalFeatures))
        # replace = False , 不重覆取
        idxs = np.random.choice(np.arange(0, totalFeatures), (sampleSize), replace=False)
        idxs.sort()
        data = []
        self._debug("starting sampling...")

        # 將挑出來的 feature 存到 data (0, 1 是X，Y座標, 2-129 是 rootSIFT 128 bin feature)
        for i in idxs:
            data.append(db["features"][i][2:])


        self._debug("sampled {:,} features from a population of {:,}".format(len(idxs), totalFeatures))
        self._debug("clustering with k={:,}".format(numClusters))
        # initialize MiniBatchKMeans
        clt = MiniBatchKMeans(n_clusters=numClusters, random_state=randomState)
        # fit operation
        clt.fit(data)
        self._debug("cluster shape: {}".format(clt.cluster_centers_.shape))

        # close the database
        db.close()

        # return the cluster centroids
        return clt.cluster_centers_


    def _debug(self, msg, msgType="[INFO]"):
        if self.verbose:
            print("{} {} - {}".format(msgType, msg, datetime.datetime.now()))
 


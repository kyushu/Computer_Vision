# -*- coding:utf-8 -*-

from searchresult import SearchResult
import numpy as np
import datetime
import dists
import h5py

class Searcher:
    def __init__(self, redisDB, bovwDBPath, featuresDBPath, idf=None, distanceMetric=dists.chi2_distance):

        self.redisDB= redisDB
        self.idf = idf
        self.distanceMetric = distanceMetric

        self.bovwDB = h5py.File(bovwDBPath, mode="r")
        self.featuresDB = h5py.File(featuresDBPath, mode="r")

    def search(self, queryHist, numResults=10, maxCandidates=200):

        # get the start time
        startTime = datetime.datetime.now()

        # 根據 queryHist, maxCandidates 挑出 候選者的 ID
        candidateIdxs = self.buildCandidates(queryHist, maxCandidates)

        # 由小到大排序，因為 HDF5 不按照順序讀取會很慢
        candidateIdxs.sort()

        # 以 候選者 ID 從 BOVW 裡挑出 feature histogram
        hists = self.bovwDB["bovw"][candidateIdxs]
        queryHist = queryHist.toarray()
        results = {}

        if self.idf is not None:
            # weighting queryHist by tf-idf
            queryHist *= self.idf

        # loop over the histograms
        for (candidate, hist) in zip(candidateIdxs, hists):
            # weighting hist by tf-idf
            if self.idf is not None:
                hist += self.idf
            # calculate chi-squared distance
            d = self.distanceMetric(hist, queryHist)
            results[candidate] = d

        # 根據 distance 來排序 並將 image index 換成 image ID
        results = sorted([ (v, self.featuresDB["image_ids"][k], k) for (k, v) in results.items()])

        results = results[:numResults]

        return SearchResult(results, (datetime.datetime.now() - startTime).total_seconds())


    def buildCandidates(self, hist, maxCandidates):
        # initialize the redis pipeline
        p = self.redisDB.pipeline()

        # loop over the column of (sparse) matrix
        for i in hist.col:
            # 從 redis 讀出
            p.lrange("vw:{}".format(i), 0, -1)

        # 執行 redis pipeline
        pipelineResults = p.execute()
        candidates = []

        # loop over the pipeline results
        for results in pipelineResults:
            # 將 results 裡的值轉成 int
            results = [int(r) for r in results]
            # extend 是將某個 list 加到這個 list 的後面
            # append 是將某個 元素 加到這個 list 的後面
            candidates.extend(results)

        (imageIdxs, counts) = np.unique(candidates, return_counts=True)
        imageIdxs = [i for (c, i) in sorted(zip(counts, imageIdxs), reverse=True)]

        return imageIdxs[:maxCandidates]

    def finish(self):
        self.bovwDB.close()
        self.featuresDB.close()






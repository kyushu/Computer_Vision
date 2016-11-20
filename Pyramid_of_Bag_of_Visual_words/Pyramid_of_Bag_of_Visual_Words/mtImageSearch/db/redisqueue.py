# -*- coding:utf-8 -*-
import numpy as np

class RedisQueue:
    def __init__(self, redisDB):
        self.redisDB = redisDB

    def add(self, imageIdx, hist):
        # initialize the redis pipline
        p = self.redisDB.pipeline()

        # loop over all non-zero entries for the histogram
        for i in np.where(hist > 0)[0]:

            # if i == 990:
            #     test = np.where(hist > 0)[0]
            #     print("hist.shape: {}".format(hist.shape))
            #     print("hist: {}".format(hist))
            #     print("test.shape: {}".format(test.shape))
            #     print("hist[0]: {}".format(test))

            # 將 "vw": value 加到 imageIdx 這個 list 的最後一個 (Right push)
            # 如果 imageIdx 這個 list 不存在，就先建立再存
            p.rpush("vw:{}".format(i), imageIdx)

        # execute the pipline
        p.execute()

    def finish(self):
        self.redisDB.save()


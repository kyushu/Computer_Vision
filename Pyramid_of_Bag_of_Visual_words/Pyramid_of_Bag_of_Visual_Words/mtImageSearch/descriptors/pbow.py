# -*- coding:utf-8 -*-

'''
weight : w = 1 / 2**(N-L+1)
N: the total number of levels in the pyramid
L: the current level of the pyramid
'''

from scipy import sparse
import numpy as np

class PBOW:
    def __init__(self, bovw, numLevels=2):
        self.bovw= bovw
        self.numLevels=numLevels

    def describe(self, imageWidth, imageHeight, kps, features):
        # 產生一個跟 imageHeight, imageWidth 大小的 mask array
        kpMask = np.zeros((imageHeight, imageWidth), dtype="int")
        concatHist = None

        # 在 mask array 對應到 keypoin 的位置填上 index
        for (i, (x, y)) in enumerate(kps):
            kpMask[y, x] = i + 1

        # 根據 Pyramid Level 計算 sub region 的權重
        # 從 numLevels 到 0, step = -1
        for level in np.arange(self.numLevels, -1, -1):
            # 根據 level 決定切成幾等分 sub region
            numParts = 2 ** level
            # 根據 level 決定 sub regiion 的 weighting
            weight = 1.0 / (2**(self.numLevels - level + 1))
            if level == 0:
                weigjht = 1.0 / (2 ** self.numLevels)


            # 從 (imageWidth / numParts) 到 imageWidth 平均切成 numParts 份
            X = np.linspace(imageWidth / numParts, imageWidth, numParts)
            Y = np.linspace(imageHeight / numParts, imageHeight, numParts)
            xParts = np.hstack([[0], X]).astype("int")
            yParts = np.hstack([[0], Y]).astype("int")


            for i in np.arange(1, len(xParts)):
                for j in np.arange(1, len(yParts)):
                    (startX, endX) = (xParts[i - 1], xParts[i])
                    (startY, endY) = (yParts[j - 1], yParts[j])
                    # 這裡要看
                    # print("np.unique(kpMask[startY:endY, startX:endX])[1:]: {}".format( np.unique(kpMask[startY:endY, startX:endX])[1:] ))

                    idxs = np.unique(kpMask[startY:endY, startX:endX])[1:] -1
                    hist = sparse.csr_matrix((1, self.bovw.codebook.shape[0]), dtype="float")

                    if len(features[idxs]) > 0:
                        hist = self.bovw.describe(features[idxs])
                        hist = weight * (hist / hist.sum())

                    if concatHist is None:
                        concatHist = hist
                    else :
                        concatHist = sparse.hstack([concatHist,hist])

        return concatHist


    '''
    將 圖片分成 Layer(L) = 2 的 pyramid
    L0: 整張圖片 (4**0)
    L1: 將圖片切成 4(4**1) 等分 cell
    L2: 將圖片切成 16(4**2) 等分 cell
    所以 cell 的總數 = ∑ 4**l, where l = 0 ~ L  =  1/3 * [ (4**(L+1)) - 1 ] (等比級數和)
    '''
    @staticmethod
    def featureDim(numClusters, numLevels):
        return int(round(numClusters * (1 / 3.0) * ((4**(numLevels + 1)) - 1)))

# -*- coding:utf-8 -*-

'''

'''

from sklearn.metrics import pairwise
from scipy.sparse import csr_matrix
import numpy as np

class BagOfVisualWords:
    def __init__(self, codebook, sparse=True):
        self.codebook = codebook
        self.sparse = sparse


    def describe(self, features):
        # 取所有的 feature 跟 codebook 裡的 feature 兩兩之間的距離
        D = pairwise.euclidean_distances(features, Y=self.codebook)
        
        # D[i, j] = features[i] array 對 self.codebook[j] 之間的距離
        # np.argmin(D, axis=1): 返回的是 D 的每個 row 的 column_index with minimum distance
        # 也就是 features[i] 對 codebook 裡每個 feature 的最小距離
        (words, counts) = np.unique(np.argmin(D, axis=1), return_counts=True)
        # 如果是 Sparse-Matrix, 則用 CSR_matrix 方式儲存(一種有效率儲存 saprse-matrix 的方式)
        if self.sparse:
            hist = csr_matrix((counts, (np.zeros((len(words),)), words)), 
                shape=(1, len(self.codebook)), dtype="float")
        else:
            hist = np.zeros((len(self.codebook),), dtype="float")
            hist[words] = counts

        return hist

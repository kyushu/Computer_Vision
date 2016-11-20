# -*- coding:utf-8 -*-

'''
Purpose:
    計算 features 與 codebook 裡的 vocabulary 相近的分數
'''

from sklearn.metrics import pairwise
from scipy.sparse import csr_matrix
import numpy as np

class BagOfVisualWords:
    def __init__(self, codebook, sparse=True):
        self.codebook = codebook
        self.sparse = sparse

    '''
    describe: 計算 features 與 codebook 裡的 vocabulary 相近的分數
    for example:
    vocabulary = [[ 0.37454012  0.95071431  0.73199394  0.59865848  0.15601864  0.15599452]
                 [ 0.05808361  0.86617615  0.60111501  0.70807258  0.02058449  0.96990985]
                 [ 0.83244264  0.21233911  0.18182497  0.18340451  0.30424224  0.52475643]]

    vocabulary 的每一個 row 代表 一個 feature cluster 的中心(centroid)
    假設
    vocabulary[0:] ＝ v0
    vocabulary[1:] ＝ v1
    vocabulary[2:] ＝ v2
    features =  [[ 0.43194502  0.29122914  0.61185289  0.13949386  0.29214465  0.36636184]
                 [ 0.45606998  0.78517596  0.19967378  0.51423444  0.59241457  0.04645041]
                 [ 0.60754485  0.17052412  0.06505159  0.94888554  0.96563203  0.80839735]
                 [ 0.30461377  0.09767211  0.68423303  0.44015249  0.12203823  0.49517691]
                 [ 0.03438852  0.9093204   0.25877998  0.66252228  0.31171108  0.52006802]
                 [ 0.54671028  0.18485446  0.96958463  0.77513282  0.93949894  0.89482735]
                 [ 0.59789998  0.92187424  0.0884925   0.19598286  0.04522729  0.32533033]
                 [ 0.38867729  0.27134903  0.82873751  0.35675333  0.28093451  0.54269608]
                 [ 0.14092422  0.80219698  0.07455064  0.98688694  0.77224477  0.19871568]
                 [ 0.00552212  0.81546143  0.70685734  0.72900717  0.77127035  0.07404465]]

    features 的每一個 row 代表這個影像的 feature vector, 總共有 10 個 feature vector
    然後去計算 feature[0:], feature[1:], ..., feature[9:] 每個 feature vector 對 vocabulary 的
    feature cluster 中心的 最小距離並計算其分數

    簡稱 feature[0:] = f0, feature[1:] = f1
    假設
    f0 與 v2 距離最近， v2_score += 1
    f1 與 v1 距離最近， v1_score += 1
    f2 與 v1 距離最近， v1_score += 1
    f3 與 v2 距離最近， v2_score += 1
    f4 與 v2 距離最近， v2_score += 1
    f5 與 v1 距離最近， v1_score += 1
    f6 與 v2 距離最近， v2_score += 1
    f7 與 v1 距離最近， v1_score += 1
    f8 與 v1 距離最近， v1_score += 1
    f9 與 v1 距離最近， v1_score += 1

    計算結果
    v0_score = 0
    v1_score = 6
    v2_score = 4
    hist = [0, 6, 4]

    因為 hist 有可能是 Sparse-Matrix 如上所示， 所以可以進一步用 csr_matrix 來儲存
    '''
    def describe(self, features):
        # 取所有的 feature 跟 codebook 裡的 feature 兩兩之間的距離
        D = pairwise.euclidean_distances(features, Y=self.codebook)

        '''
        D[i, j] = features[i] array 對 self.codebook[j] 之間的距離
        np.argmin(D, axis=1): 返回的是 D 的每個 row 的 column_index with minimum distance
        也就是 features[i] 對 codebook 裡每個 feature 的最小距離

        假設 features = 3 feature vector
            vocabulary of book = 2 cluster center
        D = [[1,2]
             [5,2]
             [9,6]]
        所以 f0 與 v0 的距離最小
        所以 f1 與 v1 的距離最小
        所以 f2 與 v1 的距離最小
        np.argmin(D, axis=1) = [0, 1, 1] (代表 clumn index = vocabulary index)

        (words, counts) = np.unique(np.argmin(D, axis=1), return_counts=True)
        words  = [0, 1] (index array)
        counts = [1, 2] (index 出現的次數)
        '''
        (words, counts) = np.unique(np.argmin(D, axis=1), return_counts=True)

        # 如果是 Sparse-Matrix, 則用 CSR_matrix 方式儲存(一種有效率儲存 saprse-matrix 的方式)
        if self.sparse:
            hist = csr_matrix((counts, (np.zeros((len(words),)), words)),
                shape=(1, len(self.codebook)), dtype="float")
        else:
            # 建立一個跟 codebook 一樣大小的 array 來存這個 feature 對 codebook 裡每個 vocabulary 的分數
            hist = np.zeros((len(self.codebook),), dtype="float")
            hist[words] = counts

        return hist

# -*- coding: utf-8 -*-

'''
模擬如何建構 BOVW (Bag Of Visual Word) 的流程

vocab : 經由 clutering 一堆 image features 所得到的 visual word
features : 某個 image 的 features
'''

from __future__ import print_function
from mtImageSearch.ir import BagOfVisualWords
from sklearn.metrics import pairwise
import numpy as np

'''
產生 假的 vocab 跟  image features
'''
# np.random.seed(42)
# vocab = np.random.uniform(size=(3, 6))
# features = np.random.uniform(size=(10, 6))
# np.random.seed(84)
# vocab = np.random.uniform(size=(3, 36))
# features = np.random.uniform(size=(100, 36))
np.random.seed(42)
vocab = np.random.uniform(size=(5, 36))
features = np.random.uniform(size=(500, 36))


print("[INFO] vocabulary:\n{}\n".format(vocab))
print("[INFO] features:\n{}\n".format(features))


'''
vocab: 假設已存在一個 bag of visual word 包含 3 個 vocabulary for image (也就是 feature vector)
features: 從某張圖片取出的 feature vectors

將每個 feature 跟 vacab 裡的 3 個 vocabulary feature vector 做 distance 
並記錄每個 feature 與 vocab 裡的那個 vocabluary featue 最接近
'''
# sklearn.metrics 的 pairwise 可以 
# 1. vector 對 array
# 2. array 對 array 
# 這裡示範使用 for loop 去算出 feature vector 跟 feature vector 兩兩之間的距離
# hist = np.zeros((vocab.shape[0],), dtype="int32")
# for (i, f) in enumerate(features):
#     # 取圖片畫每個 feature vector 跟 vocab 裡的 feature vector 兩兩之間的距離
#     D = pairwise.euclidean_distances(f.reshape(1, -1), Y=vocab)
    
#     # 取得 最小 Distance 的 index
#     j = np.argmin(D)

#     print("[INFO] Closest visual word to feature #{}: {}".format(i, j))
#     # 在 hist 對應的 index 的值加 1 (量化這張圖片裡每個 feature )
#     hist[j] += 1
#     print("[INFO] Updated histogram: {}".format(hist))


# BagOfVisualWords 使用 sklearn.metrics 的 pairwise 對 target feature array 跟 Vocab 裡的
# feature array 算兩兩之間的距離，比使用 for loop 更快
bovw = BagOfVisualWords(vocab, sparse=False)
hist = bovw.describe(features)
print("[INFO] BOVW histogram: {}".format(hist))

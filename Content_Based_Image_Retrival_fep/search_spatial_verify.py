# -*- coding:utf-8 -*-

'''
usage:
    python search_spatial_verify.py --dataset ../ukbench_samples --features-db output/features.hdf5 --bovw-db output/bovw.hdf5 --codebook output/vocab.cpickle --idf output/idf.cpickle --relevant ../ukbench_samples/relevant.json --query ../ukbench_samples/ukbench00258.jpg
'''


from __future__ import print_function
from mtImageSearch.descriptors import DetectAndDescribe
from mtImageSearch.descriptors import RootSIFT
from mtImageSearch.ir import BagOfVisualWords
from mtImageSearch.ir import SpatialVerifier
from mtImageSearch.ir import Searcher
from mtImageSearch import ResultsMontage
from scipy.spatial import distance
from redis import Redis
import argparse
import cPickle
import imutils
import json
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to the directory of indexed images")
ap.add_argument("-f", "--features-db", required=True, help="Path to the features database")
ap.add_argument("-b", "--bovw-db", required=True, help="Path to the bag-of-visual-words database")
ap.add_argument("-c", "--codebook", required=True, help="Path to the codebook")
ap.add_argument("-i", "--idf", required=True, help="Path to inverted document frequencies array")
ap.add_argument("-r", "--relevant", required= True, help="Path to relevant dictionary")
ap.add_argument("-q", "--query", required=True, help="Path to the query image")
args = vars(ap.parse_args())

# 1. initialize detector and descriptor
detector = cv2.FeatureDetector_create("SURF")
descriptor = RootSIFT()
dad = DetectAndDescribe(detector, descriptor)

# 2. initialize bovw and idf
idf = cPickle.loads(open(args["idf"]).read())
vocab = cPickle.loads(open(args["codebook"]).read())
bovw = BagOfVisualWords(vocab)


# 3. Load the relevant queries dictionary
relevant = json.loads(open(args["relevant"]).read())
queryFilename = args["query"][args["query"].rfind("/") + 1:]
queryRelevant = relevant[queryFilename]

# 4. Load the query image, resize it and convert its color space to grayscale
queryImage = cv2.imread(args["query"])
cv2.imshow("Query", imutils.resize(queryImage, width=320))
queryImage = imutils.resize(queryImage, width=320)
queryImage = cv2.cvtColor(queryImage, cv2.COLOR_BGR2GRAY)

# 5-1. extract features of query image
(queryKps, queryDescs) = dad.describe(queryImage)
# 5-2. construct a bag-of-visual-words of extracted features
# 預設 BagOfVisualWords 回傳 hist 的格式是 scipy.sparse.csr_matrix 的 sparse matrix
# csr_matrix.tocoo() : 將 csr matrix 轉成 Coordinate Format (COO) 格式並回傳
queryHist = bovw.describe(queryDescs).tocoo()

# 6. connect to redis and pefrorm search
redisDB = Redis(host="localhost", port=6379, db=0)
searcher = Searcher(redisDB, args["bovw_db"], args["features_db"], idf=idf, distanceMetric=distance.cosine)
sr = searcher.search(queryHist, numResults=20)
print("[INFO] search took: {:.2f}s".format(sr.search_time))


# 7. spatially verify the results
spatialVerifier = SpatialVerifier(args["features_db"], idf, vocab)
sv = spatialVerifier.rerank(queryKps, queryDescs, sr, numResults=20)
print("[INFO] spatial verification took: {:.2f}s".format(sv.search_time))

montage = ResultsMontage((249, 320), 5, 20)
for(i, (score, resultID, resultIdx)) in enumerate(sv.results):
    print("[RESULT] {result_num}. {result} = {score:.2f}".format(result_num=i + 1, result=resultID, score=score))

    result = cv2.imread("{}/{}".format(args["dataset"], resultID))
    montage.addResult(result, text="#{}".format(i + 1), highlight=resultID in queryRelevant)

cv2.imshow("Results", imutils.resize(montage.montage, height=700))
cv2.waitKey(0)
searcher.finish()
spatialVerifier.finish()





'''
1. Coordinate Format (COO)
    是一种坐標形式的稀疏矩陣。采用三个數组row、col和data保存非零元素的信息，这三个數组的長度相同，row保存元素的行，col保存元素的列，data保存元素的值。存儲的主要优点是靈活、簡單，僅存儲非零元素以及每个非零元素的坐標。但是COO不支持元素的存取和增删，一旦創建之后，除了將之轉换成其它格式的矩陣，几乎無法對其做任何操作和矩陣運算。
    COO使用3个數组进行存儲：values,rows, andcolumn。
    其中
    數组values: 實數或複數數据，包括矩陣中的非零元素，顺序任意。
    數组rows: 數据所處的行。
    數组columns: 數据所處的列。
    参數：矩陣中非零元素的數量 nnz，3个數组的長度均为nnz.

2. Diagonal Storage Format (DIA)
    如果稀疏矩陣有僅包含非0元素的對角线，则對角存儲格式(DIA)可以减少非0元素定位的信息量。这种存儲格式對有限元素或者有限差分离散化的矩陣尤其有效。
    DIA通过两个數组确定： values、distance。
    其中
    values：對角线元素的值；
    distance：第i个distance是当前第i个對角线和主對角线的距离。

3. Compressed Sparse Row Format (CSR)
    压缩稀疏行格式(CSR)通过四个數组确定： values,columns, pointerB, pointerE.
    其中
    數组values：是一个實（複）數，包含矩陣A中的非0元，以行优先的形式保存；數组columns：第i个整型元素代表矩陣A中第i列；
    數组pointerB ：第j个整型元素给出矩陣A行j中第一个非0元的位置，等价于pointerB(j) -pointerB(1)+1 ；
    數组pointerE：第j个整型元素给出矩陣A第j行最后一个非0元的位置，等价于pointerE(j)-pointerB(1)。
'''

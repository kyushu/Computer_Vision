# -*- coding: utf-8 -*-

'''
usage:
    python extract_bovw.py --features-db output/features.hdf5 --codebook output/vocab.cpickle --bovw-db output/bovw.hdf5 --idf output/idf.cpickle
'''

from mtImageSearch.ir import BagOfVisualWords
from mtImageSearch.indexer import BOVWIndexer
import argparse
import cPickle
import h5py

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--features-db", required=True, help="Path to the features database")
ap.add_argument("-c", "--codebook", required=True, help="Path to the codebook")
ap.add_argument("-b", "--bovw-db", required=True, help="Path to where the bag-of-visual-words database will be stored")
ap.add_argument("-d", "--idf", required=True, help="Path to inverse document frequency counts will be stored")
ap.add_argument("-s", "--max-buffer-size", type=int, default=500, help="Maximum buffer size for # of features to be stored in memory")
args = vars(ap.parse_args())

#
# 1. 初始化 Bag-Of-Visual-Word
#
# 將 codebook vocabulary 從 cPickle file 裡取出
vocab = cPickle.loads(open(args["codebook"]).read())
# 並以 codebook vocabulary 初始化 bag-of-visual-words
# bovw 用來將 target image 的 feature vectors 與 vocab 裡的 feature vectors 做 clustering
bovw = BagOfVisualWords(vocab)


#
# 2. 被始化 Bag-Of-Visual-Word Indexer
#
# 從 HDF5 取出 features database 並初始化 bag-of-visual-words indexer
# features_db 包含了 3 個 dataset: "image_ids", "index", "features"
featuresDB = h5py.File(args["features_db"], mode="r")
# bovw.codebook.shape[0]    : histogram 的個數 ＝ 使用 k-means 所產生出來的 cluster center 的個數
# args["bovw_db"]           : 輸出 bovw_db HDF5 database 的路徑
# estNumImages              : 預估 database 裡 image 的數量
# maxBufferSize             : The maximum number of BOVW histograms to be stored in memory prior to
#                               writing them to disk
bi = BOVWIndexer(bovw.codebook.shape[0], args["bovw_db"],
                estNumImages=featuresDB["image_ids"].shape[0],
                maxBufferSize=args["max_buffer_size"])


#
# 3. 建立影像的 Bag-Of-Visual-Word
#
for (i, (imageID, offset)) in enumerate(zip(featuresDB["image_ids"], featuresDB["index"])):

    if i > 0 and i % 10 == 0:
        bi._debug("processed {} images".format(i), msgType="[PROGRESS")

    # offset[0]: start
    # offset[1]: end
    # 從 featuresDB 的 offset 起點到 offset 的終點 取出 feature vectors (row)
    # 而每個 fature vectors 的 0, 1 是 keypoint 的 (x, y)              (column)
    features = featuresDB["features"][offset[0]: offset[1]][:, 2:]
    # 將 image 的 fature vector 與 codebook 裡的 feature vector 做 distance compare
    # 並將結果量化為 bag-of-visual-word histogram (被歸類為某一個 cluster 的次數)
    hist = bovw.describe(features)

    # 加入 bag-of-visual-word
    bi.add(hist)


#
# 4. close the features database and finish the indexing process
#
featuresDB.close()
bi.finish()


#
# 5. dump the inverse document frequency counts to file
#
f = open(args["idf"], "w")
f.write(cPickle.dumps(bi.df(method="idf")))
f.close()

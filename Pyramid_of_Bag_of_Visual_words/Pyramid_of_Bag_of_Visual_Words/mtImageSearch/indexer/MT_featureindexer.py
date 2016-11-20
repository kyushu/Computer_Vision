# -*- coding: UTF-8 -*-

'''
HDF5:
    HDF5 file are meant to store multiple groups and datasets.
        A group is a container-like structure which can hold datasets (and other groups).

    A dataset, however, is meant to store a multi-dimensional array of a
        homogenous data type — essentially a NumPy matrix.

    HDF5 datasets support additional features such as compression,
        error detection, and chunked I/O.

    we’ll be treating HDF5 datasets as large NumPy arrays with shape ,
        size , and dtype  attributes.

    HDF5 對於做 NumPy-like slice operation 是很快，不過對於 random access 就很慢了
'''
'''
For CBIR we'll need three datasets (which are essentially NumPy arrays)
    For instance
    1. image_ids: image UUID
        0   : uk000000.jpg
        1   : uk000001.jpg
        ...
        999 : uk00999.jpg

    2. f_index: feature index(start, end) corresponds to image
        0   : 0, 651
        1   : 651, 1386
        ...
        999 : 475280, 475589

    3. features: feature vector
        0   : 175, 45, 0.07
        1   : 123, 57, 0.07
        ...
        999 : 212, 99, 0.02

    image_ids : shape(N), 存 N 個 image unique id
    f_index   : shape(N, 2), 這兩個 integer 是 features 的 start 跟 end index
                第一個 f_index 對應到 image＿ids 的第一個 image_id
    features  : shape(M, 130)
                    M   : 是從 N 個 image 取出的 M 個 feature vectors
                    130 : 因為若使用 RootSIFT，一個 feature vector 有 128-dim
                            再加上 keypoint 的座標 (x, y)，所以需要儲存一個完整的
                            feature vector 要 130-dim.

    以 uk000000.jpg 為例
    在 image_ids[0] 的 值是 uk000000.jpg
    在 f_index[0] 的值為 (0, 651)
    這代表 uk000000.jpg 取出的 feature vector 有 651 - 0 = 651 個
    而這些 feature vector 在 features 裡的 [0, 651] (不包含第 651)

'''

from baseindexer import BaseIndexer
import numpy as np
import h5py

# 繼承自 BaseIndexer
class FeatureIndexer(BaseIndexer):

    def __init__(self, dbPath, estNumImages=500, maxBufferSize=50000, dbResizeFactor=2,
        verbose=True):

        # 呼叫父類別來初始化
        super(FeatureIndexer, self).__init__(dbPath, estNumImages=estNumImages,
            maxBufferSize=maxBufferSize, dbResizeFactor=dbResizeFactor,
            verbose=verbose)

        # open the HDF5 database file
        self.db = h5py.File(self.dbPath, mode="w")
        # initialoze the dataset within the group
        # imageIDDB 存 imageID
        self.imageIDDB = None
        # indexDB 存 image 的feature 在 featuresDB 裡 起點跟終點的 index
        self.indexDB = None
        # featuresDB 存 image 的 feature vectro , 格式為 [keypoints, features]
        self.featuresDB = None

        # imageDimsDB 存 image 的 寛高，主要是應用在 pyramid of bag of visual words 這個 process 使用
        self.imageDimsDB = None

        # initialize buffers
        self.imageIDBuffer = []
        self.indexBuffer = []
        self.featuresBuffer = None

        self.imageDimsBuffer = []

        #
        self.totalFeatures = 0
        # idxs 的 index: current row (i.e. the next empty row)
        #                   in both the image_ids  and index  datasets
        #                   對應到 HDF5 裡的 imageIDDB 的 index (也等於 indexDB 的 index)
        # idxs 的 features : the next empty row in the features  dataset
        #                   對應到 HDF5 裡的 featuresDB 的 index
        # 因為初始化，所以 index 跟 features 都是從 0 開始

        # 假設 buffer 容量為 50000, 一開始 idxs["features"] = 0, 接著輸入了 10000 筆
        # 則 featuresBuffer 的個數 為 10000，totalFeatures = 10000
        # 接著再輸入 50000，則 featuresBuffer 的個數 ＝ totalFeatures = 60000
        # 因為 60000 筆超出了 featuresBuffer 的容量，所以要將資料存入 HDF5 database file
        # 將 60000 筆存入 HDF5 database file 後， 將 featuresBuffer 清空，同時
        # totalFeatures 也重置為 0
        # idx["features"] 更新為 60000
        self.idxs = {"index": 0, "features": 0}


    def add(self, imageID, imageDim, kps, features):
        # 假設已有 50000 存到 HDF5 database file 了
        # featuresBuffer 也存了 20000 筆
        # 這時 idxs["features"] = 50000 (HDF5 database featuresDB 目前的個數)
        # totalFeatures = featureBuffer 裡的個數 ＝ 20000
        # 所以起點是 idxs["features"] + totalFeatures
        start = self.idxs["features"] + self.totalFeatures
        end = start + len(features)

        # 將 imageID 加到 imageIDBuffer
        self.imageIDBuffer.append(imageID)

        # 先將這個 image 的 keypoint 跟 feature vector 以 column 的方式並排
        # 也就是           column 0  | column 1
        # new vector  =  [keypoints,   features] (一個 keypoint 對應 一個 feature vector)
        #             =  [keypoint.x, keypoint.y, feature_1, feature_2, ...., feature_127]
        # 再將這個 new vector 跟 原本在 featureBuffer 裡的 [kps, features] 做 row-wis 的方式疊加起來
        self.featuresBuffer = BaseIndexer.featureStack(np.hstack([kps, features]),
            self.featuresBuffer)

        # 將這個 image 的 feature "start" and "end" index 存到 indexBuffer
        self.indexBuffer.append((start, end))

        # 將這個 image 的 shape 存到 imageDimsBuffer
        self.imageDimsBuffer.append(imageDim)

        # 將這個 image 的 feature 的個數 累加到 totalFeatures
        self.totalFeatures += len(features)

        # 如果 totalFeature 超過了 buffer 容量，就將 buffer 裡的資料存入 HDF5 database file
        if self.totalFeatures >= self.maxBufferSize:
            # 如果還沒建立就先建立
            if None in (self.imageIDDB, self.indexDB, self.featuresDB):
                self._debug("initial buffer full")
                self._createDatasets()
            #
            self._writeBuffers()


    def _createDatasets(self):
        # 計算一下image 的個數跟 features 的個數的比例，來推算平均一個 image 會產出幾個 features
        avgFeatures = self.totalFeatures / float(len(self.imageIDBuffer))
        # 再將這個 平均值乘上 estNumImages 就是要建立的 featuresDB 的大小
        approxFeatures = int(avgFeatures * self.estNumImages)

        # 取得我們存入 featuresBuffer 的 features vector 的 column 數
        fvectorSize = self.featuresBuffer.shape[1]

        # 初始化 HDF5 dataset
        # 如果一開始沒有 define "maxshape"，之後就不能 resize dataset
        # 你可以定義 maxshape=(None, 3)，因為將 row 的 max-shape 設為 "None"，column 設為 "3"
        # 這樣之後就可以對 dataset 的 "row" 做 resize，而 column 則固定為 "3"
        self._debug("creating datasets...")

        #
        # create_dataset(name, shape=None, dtype=None, data=None, **kwds)
        # create 出來旳 dataset 是 numpy.ndaray
        #
        self.imageIDDB = self.db.create_dataset("image_ids", (self.estNumImages,),
            maxshape=(None,), dtype=h5py.special_dtype(vlen=unicode))
        self.indexDB = self.db.create_dataset("index", (self.estNumImages, 2),
            maxshape=(None, 2), dtype="int")
        self.featuresDB = self.db.create_dataset("features", (approxFeatures, fvectorSize),
            maxshape=(None, fvectorSize), dtype="float")

        self.imageDimsDB = self.db.create_dataset("image_dims", (self.estNumImages, 2),
            maxshape=(None, 2), dtype="int")

    def _writeBuffers(self):
        # write the buffer to disk (寫到 HDF5 file)
        self._writeBuffer(self.imageIDDB, "image_ids", self.imageIDBuffer, "index")
        self._writeBuffer(self.indexDB, "index", self.indexBuffer, "index")
        self._writeBuffer(self.featuresDB, "features", self.featuresBuffer, "features")

        self._writeBuffer(self.imageDimsDB, "image_dims", self.imageDimsBuffer, "index")

        # increment the indexes
        self.idxs["index"] += len(self.imageIDBuffer)
        self.idxs["features"] += self.totalFeatures

        # reset the buffer and feature counts
        self.imageIDBuffer = []
        self.indexBuffer = []
        self.featuresBuffer = None
        self.totalFeatures = 0


    def finish(self):
        # if the database have not been initialized, then the original buffer were never filled up
        if None in (self.imageIDDB, self.indexDB, self.featuresDB):
            self._debug("minimum init buffer not reached", msgType=["WARN"])
            self._createDatasets()

        # write any unempty buffers to file
        self._debug("writing un-empty buffers...")
        self._writeBuffers()

        # compact datasets
        self._debug("compacting datasets...")
        self._resizeDataset(self.imageIDDB, "image_ids", finished=self.idxs["index"])
        self._resizeDataset(self.indexDB, "index", finished=self.idxs["index"])
        self._resizeDataset(self.featuresDB, "features", finished=self.idxs["features"])

        self._resizeDataset(self.imageDimsDB, "image_dims", finished=self.idxs["index"])

        self.db.close()



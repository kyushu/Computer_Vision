
# -*- coding: UTF-8 -*-

from __future__ import print_function
import numpy as np
import datetime

# 當你定義一個類別時
#  class abc:  就等於  class abc(object)
# 所以當你建立一個新類別時，都會繼承自 python 的 object 類別
# 而 object class 已 define 了 __init__ 這個 method
# 所以我們只是 override __init__ 而已
class BaseIndexer(object):
    def __init__(self, dbPath, estNumImages=500, maxBufferSize=50000, dbResizeFactor=2,
        verbose=True):

        # 將資料相關的資訊存下來
        self.dbPath = dbPath
        self.estNumImages = estNumImages
        self.maxBufferSize = maxBufferSize
        self.dbResizeFactor = dbResizeFactor
        self.verbose = verbose

        # initialize the indexes dictionary
        # 為了讀取給定的 row 或是 insert 新的 row 到 HDF5 dataset
        # 我們需要知道 row 的 index，而 idxs 就是用來存這些 index 的
        self.idxs = {}

    # 供子類別繼承覆寫
    def _writeBuffers(self):
        pass

    '''
    dataset     : The HDF5 dataset object that we are writing the buffer to.
    datasetName : The name of the HDF5 dataset (e.g. internal, hierarchical path).
    buf         : The buffer that will be flushed to disk.
    idxName     : The name of the key into the idx  dictionary. This variable will allow us
                    to access the current pointer into the HDF5 dataset and ensure our
                    features are written in sequential order, without overlapping each other.
    sparse      : A flag indicating whether or not the buffer is of a sparse matrix data type
    '''
    def _writeBuffer(self, dataset, datasetName, buf, idxName, sparse=False):

        # 如果是 list 資料數就用 len(buf)
        if type(buf) is list:
            end = self.idxs[idxName] + len(buf)

        # 如果不是 list 那就是 NumPy/SciPy 的 narray
        # 那資料數就用 array.shape[0] (num of rows)
        else:
            end = self.idxs[idxName] + buf.shape[0]

        # 如果計算出來的資料數大於目前 dataset 的資料數，那就 resize 這個 dataset
        if end > dataset.shape[0]:
            self._debug("triggering `{}` db resize".format(datasetName))
            self._resizeDataset(dataset, datasetName, baseSize=end)

        # 因為 HDF5 只能存 dense array
        # 所以如果這個 buf 是 sparse (稀疏矩陣)就表示存進 buffer 的時候是用 csr_matrix 的方式存進來的
        # 所以 buf = csr_matrix 的 array
        # 所以 buf.toarray() = csr_matrix.toarray() = 轉成 sparse-matrix (有零元素的 array) = dense array
        if sparse:
            buf = buf.toarray()

        # 最後將這個 buf 存入 HDF5 的檔案
        self._debug("writing `{}` buffer".format(datasetName))
        # self.idxs[idxName] 是 row 的 index (integer)
        dataset[self.idxs[idxName]: end] = buf

    '''
    dataset     : The HDF5 dataset object that we are resizing.
    dbName      : The name of the HDF5 dataset.
    baseSize    : The base size of the dataset is assumed to be the total number of rows
                    in the dataset plus the total number of entries in the buffer.
    finished    : An integer indicating whether we are finished indexing feature vectors,
                    and thus the dataset should be compacted rather than expanded.
                    This value will be 0 if we are expanding the dataset and
                     > 0 if the dataset is to be compacted.
    '''
    def _resizeDataset(self, dataset, dbName, baseSize=0, finished=0):

        # 取得目前的資料個數(number of rows)
        origSize = dataset.shape[0]

        # 如果 finished 大於 0， 那就表示所有資料都已經寫完了
        # 而 finished 就等於最後寫完全部資料的長度
        if finished > 0:
            newSize = finished

        # 如果寫rows(資料) 到 dataset 時，全部的資料還沒寫完，那就將 dataset 加大到 baseSize * dbResizeFactor
        # 預設 dbResizeFactor ＝ 2, baseSize = 目前 dataset 的容量
        # 因為 resize operation 很花費電腦效能跟時間，不要常常做 resize 的動作
        # 所以這裡才會預設一次放大兩倍
        else:
            newSize = baseSize * self.dbResizeFactor

        # 將 dataset.shap (tuple) 轉成 list (因為 tuple = 不可變的 list)
        # 為了要修改 shape[0] = 資料筆數, 所以將 dataset.shape 轉成 list
        shape = list(dataset.shape)
        # 將目前的 size(number of row) 換成 newSize
        shape[0] = newSize

        # 將 dataset resize 成新的 size (numpy.ndarray.resize(tuple or int))
        dataset.resize(tuple(shape))
        self._debug("old size of `{}` : {:,}; new size: {:,}".format(dbName, origSize, newSize))


    def _debug(self, msg, msgType="[INFO]"):
        # 如果有開 verbose 就 print debug information
        if self.verbose:
            print("{} {} - {}".format(msgType, msg, datetime.datetime.now()))

    @staticmethod
    def featureStack(array, accum=None, stackMethod=np.vstack):
        # 如果沒有 accumulated array 就建立一個
        if accum is None:
            accum = array
        # 如果有，那就 stack the arrays (預設為 np.vstack = row wise)
        else :
            accum = stackMethod([accum, array])

        return accum




# -*- coding:utf-8 -*-


from baseindexer import BaseIndexer
from scipy import sparse
import numpy as np
import h5py

class BOVWIndexer(BaseIndexer):
    def __init__(self, fvectorSize, dbPath, estNumImages=500, maxBufferSize=500, dbResizeFactor=2, verbose=True):

        super(BOVWIndexer, self).__init__(dbPath, estNumImages=estNumImages,
            maxBufferSize=maxBufferSize, dbResizeFactor=dbResizeFactor, verbose=verbose)

        # open HDF5 database for writting
        self.db = h5py.File(self.dbPath, mode="w")
        # 初始化 bovwDB dataset
        self.bovwDB = None
        # 初始化 bovwbuffer list
        self.bovwBuffer = None
        # 初始化 bovwDB 的 index
        self.idxs = {"bovw": 0}

        # 保存 bag-of-visual-words 的 feature vector 的大小
        self.fvectorSize = fvectorSize
        # 初始化 document frequency
        self._df = np.zeros((fvectorSize,), dtype="float")
        self.totalImages = 0


    def add(self, hist):
        self.bovwBuffer = BaseIndexer.featureStack(hist, self.bovwBuffer, stackMethod=sparse.vstack)
        # scipy.sparse.dia_matrix.toarray : Return a dense ndarray representation of this matrix

        self._df[np.where(hist.toarray()[0] > 0)] += 1

        # check to see if we have reached the maximum buffer size
        if self.bovwBuffer.shape[0] >= self.maxBufferSize:
            # if the databases have not been cireated yet, create them
            if self.bovwDB is None:
                self._debug("initial buffer full")
                self._createDatasets()

            # write the buffers to file
            self._writeBuffers()

    def _writeBuffers(self):
        # 只有 buffer 裡有東西時才寫入檔案
        if self.bovwBuffer is not None and self.bovwBuffer.shape[0] > 0:
            # write the BOVW buffer to file (BaseIndexer._writeBuffer)
            self._writeBuffer(self.bovwDB, "bovw", self.bovwBuffer, "bovw", sparse=True)
            # increment the index
            self.idxs["bovw"] += self.bovwBuffer.shape[0]
            # reset the buffer
            self.bovwBuffer = None


    def _createDatasets(self):
        self._debug("creating datasets...")

        #
        # create_dataset(name, shape=None, dtype=None, data=None, **kwds)
        #
        self.bovwDB = self.db.create_dataset("bovw", (self.estNumImages, self.fvectorSize), maxshape=(None, self.fvectorSize), dtype="float")


    def finish(self):

        # 如果 bovwDB 不存在，表示 bovwBuffer 是第一次被填滿
        if self.bovwDB is None:
            self._debug("minimum init buffer not reached", msgType="[WARN]")
            self._createDatasets()

        # 將 bovwBuffer 內容寫入 hdf5 file 並清除 bovwBuffer
        self._debug("Warning un-empyt buffers...")
        self._writeBuffers()

        # compact datasets (BaseIndexer._resizeDataset)
        self._debug("compacting datasets...")
        self._resizeDataset(self.bovwDB, "bovw", finished=self.idxs["bovw"])

        # 儲存 total number of images in the dataset
        self.totalImages = self.bovwDB.shape[0]
        # close the database
        self.db.close()


    def df(self, method=None):
        if method == "idf":
            # np.log: Natural logarithm, element-wise
            return np.log(self.totalImages / (1.0 + self._df))

        # 如果沒有特定的 method，就返回 df (document ferquency counts)
        return self._df


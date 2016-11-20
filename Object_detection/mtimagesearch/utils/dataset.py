#-*- coding: utf-8 -*-
#-*- coding: cp950 -*-
# 要使用中文就要加上面兩行

'''
Purpose:
    Load feature vectors and labels from a dataset residing on disk
    1. feature vectors are stored in HDF5
'''

import numpy as np
import h5py

# data          : the list of feature vectors that will be written to the HDF5 dataset

# labels        : The list of labels associated with each feature vector.
#                 The label values will be contained in {-1, 1}, where -1 indicates that the feature
#                 vector is not representative of the object we want to detect, and a value of 1
#                 indicates the feature vector is representative of the object we want to detect.

# path          : This is the path to where our HDF5 dataset will be stored on disk.

# datasetName   : The name of the dataset within the HDF5 file.

# writeMethod   : This parameter is entirely optional — it is simply the write mode of the file.
#                 We specify a value of w  here by default, indicating that the database should be
#                 opened for writing. Later in this module, we’ll supply a value of a ,
#                 allowing us to append hard-negative features to the dataset.

def dump_dataset(data, labels, path, datasetName, writeMethod= "w"):
    # open the database
    db = h5py.File(path, writeMethod)
    # create the dataset
    dataset = db.create_dataset(datasetName, (len(data), len(data[0]) + 1), dtype="float")
    # write the data and labels to the dataset
    dataset[0:len(data)] = np.c_[labels, data]
    # close the database
    db.close()


def load_dataset(path, datasetName):
    # open the database
    db = h5py.File(path, "r")
    # load the data and labels
    (labels, data) = (db[datasetName][:,0], db[datasetName][:, 1:])
    # close the database
    db.close()

    return (data, labels)

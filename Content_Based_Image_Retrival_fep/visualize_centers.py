# -*- coding: utf-8 -*-

# usage: python visualize_centers.py --dataset ../ukbench_samples/ --features-db ./output/features.hdf5 --codebook output/vocab.cpickle --output output/vw_vis

from __future__ import print_function
from mtImageSearch import ResultsMontage
from sklearn.metrics import pairwise
import numpy as np
import progressbar
import argparse
import cPickle
import h5py
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to the directory of indexed images")
ap.add_argument("-f", "--features-db", required=True, help="Path to the features database")
ap.add_argument("-c", "--codebook", required=True, help="Path to the codebook")
ap.add_argument("-o", "--output", required=True, help="Path to output directory")

args = vars(ap.parse_args())

# 載入 codebook (cPickle 格式)
vocab = cPickle.loads(open(args["codebook"]).read())
# 載入 featuresDB (hdf5 格式)
featuresDB = h5py.File(args["features_db"], mode="r")
print("[INFO] starting distance computations...")



# for instancce
# test = {i:[] for i in np.arange(0, 4)}
# test = {0: [], 1: [], 2: [], 3: []}
vis = {i:[] for i in np.arange(0, len(vocab))}
widgets = ["Comparing: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=featuresDB["image_ids"].shape[0], widgets=widgets).start()


# 1.
# 從 featuresDB 裡的 image_ids 這個 dataset 裡取出 imageID
# smallFDB = featuresDB["image_ids"][:10]
# for (i, imageID) in enumerate(smallFDB):
for (i, imageID) in enumerate(featuresDB["image_ids"]):
    # 再從 featuresDB 裡的 index 這個 dataset 取出這個 image 的 feature vectors
    (start, end) = featuresDB["index"][i]
    rows = featuresDB["features"][start:end]
    (kps, descs) = (rows[:, :2], rows[:, 2:])
    # 取出 image 的 keypoint 跟 feature vector 
    for (kp, features) in zip(kps, descs):
        # 計算 feature vector 跟 codebook 裡所有的 cluster 的 Eculidean distance 
        D= pairwise.euclidean_distances(features.reshape(1, -1), Y=vocab)[0]

        # loop 所有計算的距離
        for j in np.arange(0, len(vocab)):
            
            # 取出 vis[j] 的 value 也就是 array (初始值是 空 array)
            topResults = vis.get(j)
            
            # 將 D[j], kp, imageID 包成 tuple 存到 topResults
            topResults.append((D[j], kp, imageID))
            
            # 根據 距離 來排序，並只保留前 16 個
            topResults = sorted(topResults, key=lambda r:r[0])[:16]
            
            # 再存回去
            vis[j] = topResults
    
    pbar.update(i)

pbar.finish()
featuresDB.close()
print("[INFO] writing visualizations to file...")


# 2.
for (vwID, results) in vis.items():
    montage = ResultsMontage((64, 64), 4, 16)

    for (_, (x, y), imageID) in results:
        # print("x: {}, y: {}".format(x, y))
        p = "{}/{}".format(args["dataset"], imageID)
        image = cv2.imread(p)
        (h, w) = image.shape[:2]

        (startX, endX) = (max(0, x - 32), min(w, x + 32))
        (startY, endY) = (max(0, y - 32), min(h, y + 32))
        roi = image[startY: endY, startX:endX]

        # 把 keypoint 範圍的 content 加到 montage
        montage.addResult(roi)


    # 儲存 visualization 的圖檔
    p = "{}/vis_{}.jpg".format(args["output"], vwID)
    cv2.imwrite(p, cv2.cvtColor(montage.montage, cv2.COLOR_BGR2GRAY))

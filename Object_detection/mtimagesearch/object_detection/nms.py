# -*- coding:utf-8 -*-
from __future__ import print_function
import numpy as np

def non_max_suppression(boxes, probs, overlapThresh):

    if len(boxes) == 0:
        return []

    # if the bounding boxes are integers, convert them to floats
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    # np.argsort : return the indices that would sort this array
    # for example: a = [1,3,2,4,8,5,6,7]
    # idxs = np.argsort(a) = [0, 2, 1, 3, 5, 6, 7, 4]
    idxs = np.argsort(probs)

    # 以下的方法會找出一堆重疊的 bounding box 中 probability 最高的 bounding box
    # 若有一個 bounding box 沒有跟其它的重疊，那它也會被挑出來，而這個 bounding box 有可能會 false-positive
    print("idxs.len: {}".format(len(idxs)))
    while len(idxs) > 0:
        # get the last index in the idxs (with highest probability)
        last = len(idxs) - 1
        i = idxs[last]
        print("i={}, last={}".format(i, last))
        pick.append(i)
        print("pick:{}".format(pick))
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.maximum(x2[i], x2[idxs[:last]])
        yy2 = np.maximum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]
        # test
        # print("overlap: {}".format(overlap))
        # oltest = np.where(overlap > overlapThresh)[0]
        # print("oltest: {}".format(oltest))
        # deltest = np.concatenate(([last], oltest))
        # print("deltest: {}".format(deltest))

        #
        idxs = np.delete(idxs, np.concatenate( ([last], np.where(overlap > overlapThresh)[0] ) ) )

    return boxes[pick].astype("int")



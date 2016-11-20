
#-*- coding: utf-8 -*-　　
#-*- coding: cp950 -*-　
# 要使用中文就要加上面兩行

# usage: python explore_dims.py --conf ./conf/cars_side.json

'''
Purpose:
    這個 module 主要是要求得所要偵測物件平均的寛(width), 高(heights) 跟比例(aspect ratio)
    藉此可以決定我們的 sliding window 大小
'''


from __future__ import print_function
from mtimagesearch.utils import Conf
# 使用 scipy 的 io 來讀取 *.mat 檔案 (MATLAB file format)
from scipy import io
import numpy as np
import argparse
# glob: Unix style pathname pattern expansion
import glob


ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to the configuration file")
args = vars(ap.parse_args())

# load the configuration file and initialize the list of widths and heights
conf = Conf(args["conf"])
widths = []
heights = []

# 載入已標好物件的 bounding box 資訊並拿來計算其平均長寛跟比例
for p in glob.glob(conf["image_annotations"] + "/*.mat"):

    (y, h, x, w) = io.loadmat(p)["box_coord"][0]
    widths.append(w - x)
    heights.append(h - y)

(avgWidth, avgHeight) = (np.mean(widths), np.mean(heights))
print("[INFO] avg. width: {:.2f}".format(avgWidth))
print("[INFO] avg. height: {:.2f}".format(avgHeight))
print("[INFO] aspect ratio: {:.2f}".format(avgWidth / avgHeight))

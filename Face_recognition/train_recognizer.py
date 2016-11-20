# -*- coding:utf-8 -*-

'''
USAGE:
    python train_recognizer.py --selfies output/faces --classifier output/classifier --sample-size 500
'''

from __future__ import print_function
from mtImageSearch.face_recognition import FaceRecognizer
from imutils import encodings
import numpy as np
import argparse
import random
import glob
import cv2

# selfies     : 要新增的臉部影像檔案資料夾，
#                   這裡用的是使用 gather_selfies.py 擷取 camser 的影像並轉成 base64 的編碼儲存的檔案
# classifier  : 改訓練好的 clssifier model
# sample-size : 所需的最大樣本數

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--selfies", required=True, help="path to the selfies directory")
ap.add_argument("-c", "--classifier", required=True, help="path to the output classifier directory")
ap.add_argument("-n", "--sample-size", type=int, default=100, help="maximum sample size for each face")
args = vars(ap.parse_args())

# initialize the face recognizer and the list of labels
fr = FaceRecognizer(cv2.createLBPHFaceRecognizer(radius=1, neighbors=8, grid_x=8, grid_y=8))
labels = []

for (i, path) in enumerate(glob.glob(args["selfies"] + "/*.txt")):
    name = path[path.rfind("/") + 1:].replace(".txt", "")
    print("[INFO] training on '{}'".format(name))

    # load the face file, sample it
    # 因為這個檔案是一個純文字檔，裡面所儲存的是以一串 base64 字串代表一個臉部影像
    # 每個 base64 字串以 "換行 (new line)" 區分
    sample = open(path).read().strip().split("\n")
    sample = random.sample(sample, min(len(sample), args["sample_size"]))
    faces = []

    # 將 base64 字串轉回 binary
    for face in sample:
        faces.append(encodings.base64_decode_image(face))


    fr.train(faces, np.array([i] * len(faces)))

    labels.append(name)

fr.setLabels(labels)
fr.save(args["classifier"])




# -*- coding:utf-8 -*-

'''
usage:
    python sample_dataset.py --input ../data/caltech5 --output output/data --training-size 0.75

'''

from  imutils import paths
import argparse
import random
import shutil
import glob
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="Path to input directory of image class")
ap.add_argument("-o", "--output", required=True, help="Path to output directory to store training and testing images")
ap.add_argument("-t", "--training-size", type=float, default=0.75, help="% of image to use for training data")

args = vars(ap.parse_args())

# if directory exist, delete it
if os.path.exists(args["output"]):
    shutil.rmtree(args["output"])

# create output directory and it's sub-directory
os.makedirs(args["output"])
os.makedirs("{}/training".format(args["output"]))
os.makedirs("{}/testing".format(args["output"]))

'''
input 的資料夾有 cars, airplans, motobike, faces, guitars
隨機的從這些資料夾取出 training_size 數量的圖檔當作 training data
剩下的當作 testing data
'''
# 從 input 裡將 training_size 數量的圖檔複製到 output/training, 剩下的複製到 output/testing
for labelPath in glob.glob(args["input"] + "/*"):
    label = labelPath[labelPath.rfind("/") + 1:]
    os.makedirs("{}/training/{}".format(args["output"], label))
    os.makedirs("{}/testing/{}".format(args["output"], label))

    imagePaths = list(paths.list_images(labelPath))
    random.shuffle(imagePaths)
    i = int(len(imagePaths) * args["training_size"])

    # 只取 training_size 的圖檔來做 training data
    for imagePath in imagePaths[:i]:
        filename = imagePath[imagePath.rfind("/") + 1:]
        shutil.copy(imagePath, "{}/training/{}/{}".format(args["output"], label, filename))

    # 剩下的做 testing data
    for imagePath in imagePaths[i:]:
        filename = imagePath[imagePath.rfind("/") + 1:]
        shutil.copy(imagePath, "{}/testing/{}/{}".format(args["output"], label, filename))





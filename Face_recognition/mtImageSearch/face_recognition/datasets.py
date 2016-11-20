# -*- coding:utf-8 -*-

'''
CALTECH Faces 包含 
1. ImageData.mat 包含所有 jpg 圖檔裡的人臉位置資訊
	ImageData.mat 是一個 dictionary (key, value) 包含的 key 如下所示
	key: SubDir_Data : 包含所有 jpg 圖檔裡的人臉位置資訊
	key: __version__ : 版本
	key: __header__  : 檔案資訊
	key: __globals__ :

2. 26 個人臉圖片資料夾，每個資料夾裡包含 5 ~ 20 張不等的 jpg, 而每張圖片檔名皆為連續編號例如
	image_0004.jpg

	person_001 資料夾裡的圖片從 image_0001.jpg ~ image_0021.jpg
	person_002 資料夾裡的圖片從 image_00022.jpg ~ image_0041.jpg

	etc...
'''


# import the necessary packages
from sklearn.datasets.base import Bunch
from imutils import paths
from scipy import io
import numpy as np
import random
import cv2

def load_caltech_faces(datasetPath, min_faces=10, face_size=(47, 62), equal_samples=True,
	test_size=0.33, seed=42, flatten=False):
	# grab the image paths associated with the faces, then load the bounding box data
	imagePaths = sorted(list(paths.list_images(datasetPath)))
	# 使用 scipy.io 來載入 mat 檔(matlab 的檔案), 返回的是 numpy.ndarray
	bbData = io.loadmat("{}/ImageData.mat".format(datasetPath))
	# 將 SubDir_Data 作 Transpose
	bbData = bbData["SubDir_Data"].T

	# set the random seed, then initialize the data matrix and labels
	random.seed(seed)
	data = []
	labels = []

	# loop over the image paths
	for (i, imagePath) in enumerate(imagePaths):
		# load the image and convert it to grayscale
		image = cv2.imread(imagePath)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# grab the bounding box associated with the current image, extract the face
		# ROI, and resize it to a canonical size
		# 找到 _ 的位置後的 4位數字 字串，轉成數字再減 1 (因為編號從 0001 開始，而 array index starts from 0)
		k = int(imagePath[imagePath.rfind("_") + 1:][:4]) - 1
		(xBL, yBL, xTL, yTL, xTR, yTR, xBR, yBR) = bbData[k].astype("int")
		face = gray[yTL:yBR, xTL:xBR]
		face = cv2.resize(face, face_size)

		# check to see if the face should be flattened into a single row
		if flatten:
			# numpy.ndarray.flatten(): return a copy of the array collapsed into one dimension
			face = face.flatten()

		# update the data matrix and associated labels
		data.append(face)
		labels.append(imagePath.split("/")[-2])

	# convert the data matrix and labels list to a NumPy array
	data = np.array(data)
	labels = np.array(labels)

	# # check to see if equal samples for each face should be used
	if equal_samples:
		# initialize the list of sampled indexes
		sampledIdxs = []

		# loop over the unique labels
		for label in np.unique(labels):
			# grab the indexes into the labels array where labels equals the current
			# label
			labelIdxs = np.where(labels == label)[0]

			# only proceed if the required number of minimum faces can be met
			if len(labelIdxs) >= min_faces:
				# randomly sample the indexes for the current label, keeping only minumum
				# supplied amount, then update the list of sampled idnexes
				labelIdxs = random.sample(labelIdxs, min_faces)
				sampledIdxs.extend(labelIdxs)

		# use the sampled indexes to select the appropriate data points and labels
		random.shuffle(sampledIdxs)
		data = data[sampledIdxs]
		labels = labels[sampledIdxs]

	# compute the training and testing split index
	idxs = range(0, len(data))
	random.shuffle(idxs)
	split = int(len(idxs) * (1.0 - test_size))

	# split the data into training and testing segments
	(trainData, testData) = (data[:split], data[split:])
	(trainLabels, testLabels) = (labels[:split], labels[split:])

	# create the training and testing bunches
	training = Bunch(name="training", data=trainData, target=trainLabels)
	testing = Bunch(name="testing", data=testData, target=testLabels)

	# return a tuple of the training, testing bunches, and original labels
	return (training, testing, labels)
# -*- coding:utf8 -*-

# Usage: python eigenfaces.py --dataset ./datasets/caltech_faces/
# python eigenfaces.py --dataset ./caltech_faces/ --visualize 1

from __future__ import print_function
from mtImageSearch.face_recognition.datasets import load_caltech_faces
from mtImageSearch import ResultsMontage
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from skimage import exposure
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to CALTECH faces dataset")
ap.add_argument("-n", "--num-components", type=int, default=150, help="# of principal components")
ap.add_argument("-s", "--sample-size", type=int, default=10, help="# of example samples")
ap.add_argument("-v", "--visualize", type=int, default=-1, help="whether or not PCA components should be visualized")
args = vars(ap.parse_args())


################################################################################
# 1. Load the CALTECH Faces dataset and split it into training and testing data
################################################################################
print("[INFO] loading CALTECH Faces dataset...")
# flatten: transform "MxN" array to "1xN" array
# 因為要套用 PCA 所以這裡將 CALTECH faces dataset 裡所有的圖片 從 MxN 轉為 1xN  1 維陣列
# 預設將 image resize 成 47x62 的大小
(training, testing, names) = load_caltech_faces(args["dataset"], min_faces=21, flatten=True, test_size=0.25)



################################################################################
# 2. compute the PCA (eigenfaces) representation of the data, then project
#    the training data onto the eigenfaces subspace
################################################################################
print("[INFO] creating eigenfaces...")
# RandomizedPCA: 是一個 Approximate PCA 計算速度較快，詳細參考
# http://scikit-learn.org/stable/modules/decomposition.html
'''
PCA
1. Compute the mean ui of each column in the matrix, giving us the average pixel intensity value for every (x, y)-coordinate in the image dataset.
2. Subtract the ui from each column ci — this is called mean centering the data and is a required step when performing PCA.
3. Now that our matrix M has been mean centered, compute the covariance matrix.
4. Perform an eigenvalue decomposition on the covariance matrix to get the the eigenvalues λi and eigenvectors Xi.
5. Sort Xi by |λi|, largest to smallest.
6. Take the top N eigenvectors with the largest corresponding eigenvalue magnitude.
7. Transform the input data by projecting (i.e., taking the dot product) it onto the space created by the top N eigenvectors — these eigenvectors are called our eigenfaces.

PCA 將 image 所有的 pixel(NxM 的 array) flatten 成一個 vector 做 PCA component
再取前 150 largest eigenvalue 的 eigenvector 視為 eigenfaces
所以餵給 PCA 的 image size 不能太大，否則會做很久

eigenfaces are the most prominent deviations from the mean in our face dataset
'''
# pca.fit_transform(data) 同時包含了
# 1. 計算出 eigenvector 並只留前 150 largest eigenvalue 的 eigenvector 稱為 eigenfaces
# 2. 將 data project 到 eigenfaces space.
pca = RandomizedPCA(n_components=args["num_components"], whiten=True)
trainData = pca.fit_transform(training.data)



################################################################################
# 3. check to see if the PCA components should be visualized
################################################################################
if args["visualize"] > 0:

    # 假設 num-components ＝150，則 PCA.components = eigenVecto = eigenFace 會有 150 個
    # 這裡我們只看前 16 個

    # initialize the montage for the components
    # 每一列有 4 個，總共 16 個，每張圖大小為 47x62
    montage = ResultsMontage((62, 47), 4, 16)

    # loop over the first 16 individual components
    print("total number of components: {}".format(pca.components_.shape))
    for (i, component) in enumerate(pca.components_[:16]):
        print("component.shap: {}".format(component.shape))
        # 因為一開始載入 CALTECH Faces dataset 就將所有 image resize 成 47x62 的大小
        # 將 PCA 的 component 轉成 47x62 pixel bitmap
        component = component.reshape((62, 47))
        # 再將 component 裡的值轉為 uint8 的型態並給定範圍 0 ~ 255
        component = exposure.rescale_intensity(component, out_range=(0, 255)).astype("uint8")
        # 將 component 裡的每個 element 由 1 個值 轉成 3個值(BGR)
        component = np.dstack([component] * 3)
        montage.addResult(component)

    mean = pca.mean_.reshape((62, 47))
    mean = exposure.rescale_intensity(mean, out_range=(0, 255)).astype("uint8")
    cv2.imshow("Mean", mean)
    cv2.imshow("Component", montage.montage)
    cv2.waitKey(0)

################################################################################
# 4. Train a classifier on the eigenfaces representation (SVM)
################################################################################
print("[INFO] training classifier...")
model = SVC(kernel="rbf", C=10.0, gamma=0.001, random_state=84)
model.fit(trainData, training.target)

################################################################################
# 5. Evaluate the mode
################################################################################
print("[INFO] evaluating model...")
predictions = model.predict(pca.transform(testing.data))
print(classification_report(testing.target, predictions))


################################################################################
# 6. 隨機從 test.data 挑選出 sample＿size 個 image 看預測結果
################################################################################
for i in np.random.randint(0, high=len(testing.data), size=(args["sample_size"],)):
    # 從 testdata 取出 image, 因為一開始有做 flatten，所以現在轉回 47x62 的大小
    face = testing.data[i].reshape((62,47)).astype("uint8")
    prediction = model.predict(pca.transform(testing.data[i].reshape(1, -1)))

    print("[INFO] Prediction: {}, Actual: {}".format(prediction[0], testing.target[i]))
    face = imutils.resize(face, width=face.shape[1] *2, inter=cv2.INTER_CUBIC)
    cv2.imshow("Face", face)
    cv2.waitKey(0)



# -*- coding:utf-8 -*-

from collections import namedtuple
import cPickle
import cv2
import os

# define a data structure to encapsulate "FaceRecognizer" instance, so it can be serialized to disk
FaceRecognizerInstance = namedtuple("FaceRecognizerInstance", ["trained", "labels"])

class FaceRecognizer:
    def __init__(self, recognizer, trained=False, labels=None):
        # store the face recognizer wehther or not the face recognizer has already been trained
        # 這裡使用的是 LBP recognizer，因為相較於 Eigenface
        # 1. LBP 可以讓我們加入新的照片並更新 model(只 training 新加入的), 而 Eigenface 要 re-tain at all
        # 2. LBP is more robust to changes in orientation and lighting conditions than Eigenface
        self.recognizer = recognizer
        self.trained = trained
        # store the list of face name labels
        self.labels = labels

    def setLabels(self, labels):
        # store the face name labels
        self.labels = labels

    def setConfidenceThreshold(self, confidenceThreshold):
        # set the threshold for the classifier
        # if "chi-square distance" above the threshold will be marked as "unknow" face
        # that means we(classifier) don't know who the person in the image/video is
        self.recognizer.setDouble("threshold", confidenceThreshold)

    def train(self, data, labels):
        # if the model has not been trained, train it
        if not self.trained:
            self.recognizer.train(data, labels)
            self.trained = True
            return

        # orherwise, update the model (只 train 新的)
        self.recognizer.update(data, labels)


    def predict(self, face):
        # 因為使用是 LBP 請參考 ../../lbp_faces.py
        (prediction, confidence) = self.recognizer.predict(face)

        if prediction == -1:
            return ("Unknown", 0)

        # this "confidence" = chi-square between the face and the closest point in the data list
        return (self.labels[prediction], confidence)

    # store classifier to file
    # labels
    def save(self, basePath):
        fri = FaceRecognizerInstance(trained=self.trained, labels=self.labels)

        # 先確定 檔案是否存在，若否 則建立
        if not os.path.exists(basePath + "/classifier.model"):
            f = open(basePath + "/classifier.model", "w")
            f.close()
        # FaceRecognizer::save (存 LBPHFaceRecognizer 到檔案)
        self.recognizer.save(basePath + "/classifier.model")

        # 存這個 FaceRecognizer 到檔案
        f = open(basePath + "/fr.cpickle", "w")
        f.write(cPickle.dumps(fri))
        f.close()


    @staticmethod
    def load(basePath):
        # 從檔案載入(初始化)這個 FaceRecognizer
        fri = cPickle.loads(open(basePath + "/fr.cpickle").read())
        # 建立 LBPHFaceRecognizer
        recognizer = cv2.createLBPHFaceRecognizer()
        # 從檔案載入 已訓練的 classifier 到 LBPHFaceRecognizer
        recognizer.load(basePath + "/classifier.model")

        # 返回完整的 FaceRecognizer
        return FaceRecognizer(recognizer, trained=fri.trained, labels=fri.labels)




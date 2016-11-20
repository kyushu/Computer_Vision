
# -*- coding: utf-8 -*-

'''
將 image 排列
'''

import numpy as np
import cv2

class ResultsMontage:
    def __init__(self, imageSize, imagesPerRow, numResults):
        
        self.imageH = imageSize[0] # Height, Row
        self.imageW = imageSize[1] # Width, Colomn
        self.imagesPerRow = imagesPerRow
        
        # allocate memory for the output image
        numCols = numResults // imagesPerRow # // floor dive
        self.montage = np.zeros((numCols * self.imageH, imagesPerRow * self.imageW, 3), dtype="uint8")

        # initial the counter for the current image along with the row and column number
        self.counter = 0
        self.row = 0
        self.col = 0


    def addResult(self, image, text=None, highlight=False):
        # 由左至右，由上而下排列
        if self.counter != 0 and self.counter % self.imagesPerRow == 0:
            self.col = 0
            self.row += 1

        # cv2.resize(src, size, dst)
        # src   : input image
        # size  : (width, height)
        # dst   : resized image
        image = cv2.resize(image, (self.imageW, self.imageH))
        (startY, endY) = (self.row * self.imageH, (self.row + 1) * self.imageH)
        (startX, endX) = (self.col * self.imageW, (self.col + 1) * self.imageW)
        # print "startX: %s, endX: %s" % (startX, endX)
        # print "startY: %s, endY: %s" % (startY, endY)
        self.montage[startY: endY, startX: endX] = image

        if text is not None:
            cv2.putText(self.montage, text, (startX + 10, startY + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

        if highlight:
            cv2.rectangle(self.montage, (startX + 3, startY + 3), (endX - 3, endY - 3), (0, 255, 0), 4)

        self.col += 1
        self.counter += 1

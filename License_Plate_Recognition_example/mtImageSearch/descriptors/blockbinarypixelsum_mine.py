# -*- coding:utf-8 -*-

'''
Description:
    這裡使用的是 "Block-Binary-Pixel-Sum Descriptor"  abbreviation is BBPS
    The BBPS descriptor is actually simple, and functions by dividing an image into non-overlapping M x N pixel blocks

    1. take binary image
    2. divide image into non-overlapping M x N pixel blocks
    3. count each block non zero pixels to form feature vector
        backgorund is black(pixel = 0)
        foreground(character) is white(pixel = 255)

'''

import numpy as np
import cv2

class BlockBinaryPixelSum:
    def __init__(self, targetSize=(30, 15), blockSizes=((5, 5),)):
        # targetSize: canonical size for ROI
        self.targetSize = targetSize
        # blockSizes is a list of tuple
        self.blockSizes = blockSizes


    def describe(self, image):
        # resize image to targetSize
        image = cv2.resize(image, (self.targetSize[1], self.targetSize[0]))
        features = []

        # loop over the block sizes
        for (blockW, blockH) in self.blockSizes:
            # loop for the current block size (y, x)
            for y in xrange(0, image.shape[0], blockH):
                for x in xrange(0, image.shape[1], blockW):
                    # get ROI rect
                    roi = image[y:y + blockH, x:x + blockW]
                    # Count the Non-Zero pixels in the ROI and normalizing by the current block size
                    total = cv2.countNonZero(roi) / float(roi.shape[0] * roi.shape[1])

                    features.append(total)
        # return the feature (numpy.narray)
        return np.array(features)


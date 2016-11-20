
#  -*- coding: utf-8 -*-

import numpy as np
import cv2

'''
RootSIFT:
    RootSIFT, which allows the SIFT descriptor to be “compared” using a Hellinger kernel — but still utilizing the Euclidean distance
    RootSIFT 可以比 SIFT 有更好的精確度

    將 SIFT 改成 RootSIFT 的步驟：
        Step 1: Compute SIFT descriptors using your favorite SIFT library (such as OpenCV).
        Step 2: L1-normalize each SIFT vector.
        Step 3: Take the square root of each element in the SIFT vector.
'''


class RootSIFT:
    def __init__(self):
        # initialize the SIFT feature extractor
        self.extractor = cv2.DescriptorExtractor_create("SIFT")

    # eps = 1e-7 主要是用來避免做除法時分母等於 0 的狀況
    def compute(self, image, kps, eps=1e-7):
        # compute SIFT descriptors
        (kps, descs) = self.extractor.compute(image, kps)

        # if there are no keypoints or descriptors, return an empty tuple
        if len(kps) == 0:
            return ([], None)

        # descs = [keypoint 的個數, 128 個 histogram bin]
        # 假設 keypoint 有 660 個，則 descs = [660, 128]
        # 將 descs 的每個 keypoint 的每個 histogram 做 L1-normalize
        descs /= (descs.sum(axis=1, keepdims=True) + eps)
        descs = np.sqrt(descs)

        return (kps, descs)




'''
# cv2.DescriptorExtractor_create.compute(image, keypoints[, descriptors]) → keypoints, descriptors
image           : Image.
images          : Image set.

keypoints       : Input collection of keypoints.
                    Keypoints for which a descriptor cannot be computed are removed.
                    Sometimes new keypoints can be added, for example: SIFT duplicates
                    keypoint with several dominant orientations (for each orientation).

descriptors    : Computed descriptors. In the second variant of the method descriptors[i]
                    are descriptors computed for a keypoints[i]. Row j is the keypoints
                    (or keypoints[i]) is the descriptor for keypoint j-th keypoint.
'''

'''
numpy.sum(a, axis=None, dtype=None, out=None, keepdims=False)[source]

For instance:
a = np.array([[1,2,3],[4,5,6]])

b = np.sum(a)
b = 21

c = np.sum(a, keepdims=True)
c = array([[21]])

d = np.sum(a, axis=0, keepdims=True)
d = array([[5, 7, 9]])

e = np.sum(a, axis=1, keepdims=True)
e = array([[ 6],
           [15]])

'''

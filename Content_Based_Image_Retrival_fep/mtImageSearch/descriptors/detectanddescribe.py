
# -*- coding: utf-8 -*-

import numpy as np

class DetectAndDescribe:
    def __init__(self, detector, descriptor):
        self.detector = detector
        self.descriptor = descriptor

    def describe(self, image, useKpList=True):

        # detect keypoint in the image
        kps = self.detector.detect(image)
        
        # exract local invariant descriptors
        (kps, descs) = self.descriptor.compute(image, kps)

        # if there are no keypoints or descriptors, return None
        if len(kps) == 0:
            return (None, None)

        # check to seee if the keypoints should be converted to a Numpu array
        if useKpList:
            kps = np.int0([kp.pt for kp in kps]) # np.int0() round the value and make it integer

        return (kps, descs)


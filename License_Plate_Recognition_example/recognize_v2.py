# -*- coding:utf-8 -*-

# usage: python recognize_v2.py --images testing_lp_dataset/
#        python recognize_v2.py --images testing_lp_dataset/65aab.jpg
# testing_lp_dataset/vtbnm.jpg

from __future__ import print_function
from mtimagesearch.license_plate import LicensePlateDetector_v2
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="Path to the images to e classified")
args = vars(ap.parse_args())

for imagePath in sorted(list(paths.list_images(args["images"]))):
    image = cv2.imread(imagePath)
    print(imagePath)

    # 統一 image 的寛為 640
    if image.shape[1] > 640:
        image = imutils.resize(image, width=640)

    # initialize the license plate detector
    lpd = LicensePlateDetector_v2(image)
    # detect the license and character candidates
    plates = lpd.detect()

    # loop over the license plate regions
    for (i, (lp, lpBox)) in enumerate(plates):

        # 標示車牌的 contour
        cv2.drawContours(image, [lpBox], -1, (0, 255, 0), 2)

        # 將 candidate 由 binary image 轉為 一般 RGB image
        candidates = np.dstack([lp.candidates] * 3)

        # 將 thresh 由 binary image 轉為 一般 RGB image
        thresh = np.dstack([lp.thresh] * 3)

        # 將 plate, thresh, candidtate 垂直排列
        output = np.vstack([lp.plate, thresh, candidates])

        cv2.imshow("Plate & Candidates #{}".format(i + 1), output)

    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


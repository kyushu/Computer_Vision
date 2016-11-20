# -*- coding:utf-8 -*-

# usage: python recognize_v3.py --images testing_lp_dataset/
#        python recognize_v3.py --images testing_lp_dataset/65aab.jpg
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


print("path list:{}".format(list(paths.list_images(args["images"]))))
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
    for (lpBox, chars) in plates:
        print("chars.shape:{}".format(len(chars)))
        # loop over each character
        for (i, char) in enumerate(chars):
            # show the character
            print("char.shape:{}". format(char.shape))
            cv2.imshow("Character {}".format(i + 1), char)

    # display the output image
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


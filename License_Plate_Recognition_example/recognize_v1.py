
# -*- coding: utf-8 -*-

# usage: python recognize.py --images testing_lp_dataset/
#        python recognize.py --images testing_lp_dataset/65aab.jpg

from __future__ import print_function
# from mtimagesearch.license_plate import LicensePlateDetector
from mtimagesearch.license_plate import LicensePlateDetector_v2
from imutils import paths
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="Path to the images to be classified")
args = vars(ap.parse_args())

for imagePath in sorted(list(paths.list_images(args["images"]))):
    image = cv2.imread(imagePath)
    print(imagePath)

    if image.shape[1] > 640:
        image = imutils.resize(image, width=640)

    lpd = LicensePlateDetector_v2(image)
    plates = lpd.detect()

    # for lpBox in plates:
    for (i, (lp, lpBox)) in enumerate(plates):
        cv2.drawContours(image, [lpBox], -1, (0, 255, 0), 2)

    cv2.imshow("image", image)
    cv2.waitKey(0)


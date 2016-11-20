# -*- coding:utf-8 -*-

'''
usage:
    python recognize_v4.py --images testing_lp_dataset/ -c output/simple_char.cpickle -d output/simple_digit.cpickle

    python recognize_v4.py --images oneData/ -c output/simple_char.cpickle -d output/simple_digit.cpickle
'''


from __future__ import print_function
from mtimagesearch.license_plate import LicensePlateDetector
from mtimagesearch.descriptors import BlockBinaryPixelSum
from imutils import paths
import numpy as np
import argparse
import cPickle
import imutils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="path to the images to be classified")
ap.add_argument("-c", "--char-classifier", required=True, help="path to the output character classifier")
ap.add_argument("-d", "--digit-classifier", required=True, help="path to the output digit classifier")
args = vars(ap.parse_args())


charModel = cPickle.loads(open(args["char_classifier"]).read())
digitModel = cPickle.loads(open(args["digit_classifier"]).read())

blockSizes = ((5, 5), (5, 10), (10, 5), (10, 10))
desc = BlockBinaryPixelSum(targetSize=(30, 15), blockSizes=blockSizes)

for imagePath in sorted(list(paths.list_images(args["images"]))):
# for imagePath in sorted(list(paths.list_images(args["images"]))):
    print(imagePath[imagePath.rfind("/") + 1:])
    image = cv2.imread(imagePath)

    # resize to 640
    if image.shape[1] > 640:
        image = imutils.resize(image, width=640)

    lpd = LicensePlateDetector(image, numChars=7)
    plates = lpd.detect()


    for (lpBox, chars) in plates:
        text = ""

        for (i, char) in enumerate(chars):
            char = LicensePlateDetector.preprocessChar(char)
            features = desc.describe(char).reshape(1, -1)
            # cv2.imshow("{}".format(i), char)
            # cv2.waitKey(0)
            # 如果是前 3 個字元就用 character classifier
            if i < 3:
                prediction = charModel.predict(features)[0]
            else:
                prediction = digitModel.predict(features)[0]


            text += prediction.upper()

        # cv2.destroyAllWindows()

        # only draw the characters and bounding box if there are some characters that
        # we can display
        if len(chars) > 0:
            # compute the center of the license plate bounding box
            M = cv2.moments(np.array([lpBox]))
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # draw the license plate region and license plate text on the image
            cv2.drawContours(image, [lpBox], -1, (0, 255, 0), 2)
            cv2.putText(image, text, (cX - (cX / 5), cY - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 0, 255), 2)

    # display the output image
    cv2.imshow("image", image)
    cv2.waitKey(0)


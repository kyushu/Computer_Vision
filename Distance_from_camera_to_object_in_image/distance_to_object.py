
# -*- coding: utf-8 -*-

# usage: python distance_to_object.py --reference reference/ref_24in.jpg --ref-width-inches 4.0 --ref-distance-inches 24.0 --images images


from pyimagesearch import DistanceFinder
from imutils import paths
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-r", "--reference", required=True, help="Path to the reference image")
ap.add_argument("-w", "--ref-width-inches", required=True, type=float, help="Reference object width in inches")
ap.add_argument("-d", "--ref-distance-inches", required=True, type=float, help="Distance to reference object in inches")
ap.add_argument("-i", "--images", required=True, help="Path to the directory containing images to test")

args = vars(ap.parse_args())


refImage = cv2.imread(args["reference"])
# resize image
refIamge = imutils.resize(refImage, height=700)

# initialize the distance finder
df = DistanceFinder(args["ref_width_inches"], args["ref_distance_inches"])
# 取得拿來當 marker 的矩形
refMarker = DistanceFinder.findSquareMarker(refIamge)
# 根據
# 1. marker 在 image 裡的 pixel width
# 2. 已知的 Marker 到 camera 的距離
# 3. 已知的 Marker 真正的 width (inches)
# 4 計算當前 camera 的 focal length
df.calibrate(refMarker[2]) # refMarker[2] = width

refImage = df.draw(refIamge, refMarker, df.distance(refMarker[2]))
cv2.imshow("Reference", refImage)


for imagePath in paths.list_images(args["images"]):
    filename = imagePath[imagePath.rfind("/") + 1:]
    image = cv2.imread(imagePath)
    image = imutils.resize(image, height=700)
    print "[INFO] processing {}".format(filename)

    marker = DistanceFinder.findSquareMarker(image)

    if marker is None:
        print "[INFO] could not find marker for {}".format(filename)
        continue

    distance = df.distance(marker[2])
    print "distance = %s" % distance
    image = df.draw(image, marker, distance)
    cv2.imshow("Image", image)
    cv2.waitKey(0)


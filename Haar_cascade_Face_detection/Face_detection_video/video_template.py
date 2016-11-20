
# -*- coding:utf-8 -*-

import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="Path to the (optional) video file")
args = vars(ap.parse_args())

if not args.get("video", False):
    # cv2.VideoCapture(0) 是指第一個 webcam
    # cv2.VideoCapture(1) 是指第二個 webcam
    # 以此類推
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args["video"])

while True:
    (grabbed, frame) = camera.read()

    if args.get("video") and not grabbed:
        break

    cv2.imshow("Frame", imutils.resize(frame, width=600))
    waitkey = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

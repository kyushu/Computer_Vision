# -*- coding:utf-8 -*-

'''
usage:
    python gather_selfies.py --face-cascade cascades/haarcascade_frontalface_default.xml --output output/faces/morpheus.txt

purpose:
    利用 base64 將圖片內容轉為 base64 的字串儲成檔案

'''

# import the necessary packages
from __future__ import print_function
from mtImageSearch.face_recognition import FaceDetector
from imutils import encodings
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face-cascade", required=True, help="path to face detection cascade")
ap.add_argument("-o", "--output", required=True, help="path to output file")
ap.add_argument("-w", "--write-mode", type=str, default="a+", help="write method for the output file")
args = vars(ap.parse_args())

# initialize the face detector, boolean indicating if we are in capturing mode or not, and
# the bounding box color
fd = FaceDetector(args["face_cascade"])
captureMode = False
color = (0, 255, 0)

# grab a reference to the webcam and open the output file for writing
camera = cv2.VideoCapture(0)
f = open(args["output"], args["write_mode"])
total = 0

while True:
    (grabbed, frame) = camera.read()

    if not grabbed:
        break
    # resize image to width = 500
    frame = imutils.resize(frame, width=500)
    # convert it to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the frame and get it bounding rect
    faceRects = fd.detect(gray, scaleFactor=1.1, minNeighbors=9, minSize=(100, 100))

    # ensure that at least one face was detected.
    if len(faceRects) > 0:
        (x, y, w, h) = max(faceRects, key=lambda b:(b[2] * b[3]))

        # if in capture mode, extract the face ROI, encode it into base64 and write it to file
        if captureMode:
            face = gray[y: y + h, x: x + w].copy(order="C")
            # encodings.base64_encode_image 會將 image array 轉成一維陣列並存成
            # image one dimenson array + type + shape
            f.write("{}\n".format(encodings.base64_encode_image(face)))
            total += 1

        # draw bounding box of face on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)


    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # press "c" for capture face
    if key == ord("c"):
        if not captureMode:
            captureMode = True
            color = (0, 0, 255)
        else:
            captureMode = False
            color = (0, 255, 0)
    # press "q" to quit
    elif key == ord("q"):
        break

print("[INFO] wrote {} frames to file".format(total))
f.close()
camera.release()
cv2.destroyAllWindows()




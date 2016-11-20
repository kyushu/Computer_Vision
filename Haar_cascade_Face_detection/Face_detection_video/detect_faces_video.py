
# -*- coding: utf-8 -*-

# usage: 
# for webcam
# python detect_faces_video.py -f ../cascades/haarcascade_frontalface_default.xml 
# for video
# python detect_faces_video.py -f ../cascades/haarcascade_frontalface_default.xml -v video.mp4 

import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required=True, help="Path to where the face cascade resides")
ap.add_argument("-v", "--video", help="Path to the (optional) video file")
args = vars(ap.parse_args())


detector = cv2.CascadeClassifier(args["face"])

if not args.get("video", False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args["video"])


while True:
    (grabbed, frame) = camera.read()

    if args.get("video") and not grabbed:
        break
    # resize frame
    frame = imutils.resize(frame, width=400)
    # convert to gray-scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces
    faceRects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)

    # put rectangle on the detected face
    for (x, y, w, h) in faceRects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break



camera.release()
cv2.destroyAllWindows()



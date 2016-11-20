
# -*- coding: utf-8 -*-

import cv2

'''
# formular : F = (P * D) / W
F : Focal Length
P : The width(in pixel) of Marker Object in camera
D : The distance between Marker Object and camera
W : the Real Width of Marker Object (in inch or centimeter)
'''
class DistanceFinder:
    def __init__(self, knownWidth, knownDistance):
        self.knownWidth = knownWidth
        self.knownDistance = knownDistance

        self.focalLength = 0

    def calibrate(self, width):
        self.focalLength = (width * self.knownDistance / self.knownWidth)

    def distance(self, perceivedWidth):
        return (self.knownWidth * self.focalLength) / perceivedWidth

    # use to find the largest, approximately square object
    @staticmethod 
    def findSquareMarker(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 35, 125)

        (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # sorted 預設為由小排到大，這裡我們由大排到小，最大的 contour 應該為我們的 marker
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        markerDim = None

        for c in cnts:
            # get perimeter(周長) of contour
            peri = cv2.arcLength(c, True)
            # 看 approx_simple.py 或是 1.11.4:Contour approximation
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                (x, y, w, h) = cv2.boundingRect(approx)
                aspectRatio = w / float(h)

                if aspectRatio > 0.9 and aspectRatio < 1.1:
                    markerDim = (x, y, w, h)
                    print "markerDim.width: %s" % markerDim[2]
                    break

        return markerDim


    @staticmethod
    def draw(image, boundingBox, dist, color=(0, 255, 0), thickness=2):
        (x, y, w, h) = boundingBox
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(image, "%.2fft" % (dist / 12), (image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 3)

        return image


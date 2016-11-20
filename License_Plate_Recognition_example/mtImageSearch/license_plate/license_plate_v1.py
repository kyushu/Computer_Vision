
# -*- coding: utf-8 -*-

import numpy as np
import cv2

class LicensePlateDetector:
    def __init__(self, image, minPlateW=60, minPlateH=20):
        # set minimum wdith and height, it depends on your dataset
        self.image = image
        self.minPlateW = minPlateW
        self.minPlateH = minPlateH

    def detect(self):
        return self.detectPlates()


    def detectPlates(self):

        regions = []

        # 因為車牌的寛度比高度長，所以將 kernel 設為 (13, 5)
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))

        # 用來產生 mask 時所需的 morphology operation 的 kernel size
        # squareKernel 配合 close operation 使用，用來關閉 image 裡的一些 gap
        squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        # 使用 MORPH_BLACKHAT 來凸顯 車牌的字(black)跟車牌的背景(white) 對比
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

        # light 是 image 中比較亮的部份，會被拿來當作 mask，主要是將一些小的 gap 閉起來 (closing)
        light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKernel)
        # 再將 close 後的 image 裡
        # grayscale pixel intensity > 50 的設為 255
        # 其它的設為 0
        light = cv2.threshold(light, 50, 255, cv2.THRESH_BINARY)[1]

        # 取 gradX 主要是用來凸顯不只是只有 黑白對比的部份，還有 vertical changes in gradient 的部份
        # 用 Sobel 計算 X 軸的 gradient (image intensity ), ksize = 1 only for one axis be used
        gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=1)
        # 取絕對值
        gradX = np.absolute(gradX)
        # 取 gradX 裡 最大值與最小值
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        # 把 gradX scale 回 0 ~ 255: 
        # gradX - minVal => 以 minVal 為 0
        # maxVal - minVal = 這個區間範圍的大小，稱為 RangeX，範圍從 0 ~ RangeX
        # (gradX - minVal) / (maxVal - minVal) => 將原本的 gradX scale 到 RangeX
        # 最後再將這個 RangeX 放大到 255 倍 => [0, 255]
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

        # 
        # Sobel 的 gradient image 有很多雜訊，所以使用 Gaussian blur smoothing
        gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
        # 接著再用 Closing operation 將 gradX 裡的 一些小 gap 關起來
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
        # 最後再用 Otsu's thresholding 來動態計算 thresholding value "T"
        # pixel > T = 255
        # pixel < T = 0
        thresh= cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # erode -> dilate : 可以消除小的 blob 所以會造成 blob 跟 blob 之間的很小的連接消失
        #                   可以使 blob 閉起來
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # 把這個 thresh 自己做 bitwise_and(值不變)，再用 light 作 mask
        # light 裡大於 255 的才會顯示
        thresh = cv2.bitwise_and(thresh, thresh, mask=light)
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.erode(thresh, None, iterations=1)
        
        # cv2.imshow("thresh", thresh)
        # cv2.waitKey(0)

        # 最後拿處理好的 thresh 來取得 contour
        (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            (x, y) = cv2.boundingRect(c)[:2] 
            (w,h) = cv2.boundingRect(c)[2:] # boundingRect 返回的是 (x, y, w, h)
            aspectRatio = w / float(h)

            rect = cv2.minAreaRect(c)
            box = np.int0(cv2.cv.BoxPoints(rect))
            
            # for test 
            if aspectRatio > 2.9:         
                cv2.drawContours(self.image, [box], -1, (0, 255, 0), 2)
                cv2.putText(self.image, "%s" % (aspectRatio), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
                # print("ratio: {}, w:{}, h:{}".format(aspectRatio, w, h))
            # for test

            if (aspectRatio > 3 and aspectRatio < 6) and h > self.minPlateH and w > self.minPlateW:
                regions.append(box)
        cv2.imshow("boxed", self.image)
        cv2.waitKey(0)
        return regions


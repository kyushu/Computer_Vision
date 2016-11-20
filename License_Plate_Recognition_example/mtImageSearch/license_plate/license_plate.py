# -*- coding:utf-8 -*-


'''
Note:
    在做 License Plate Recogintion 時， Contour property 很重要
    如何定義 Character 的寛高比, solidity, Character 的高跟車牌的高的比例
    並沒有一個固定的數字，需依各種車牌來制訂
    像 數字或英文字母的 solidity > 0.15 即可
    而 國字可能 solidity > 50

members:
    init : initialize LicensePlateDetector_v2 with width, heightm, number of character, minimum character width

    detect:

'''


# import the necessary packages
from collections import namedtuple
from skimage.filters import threshold_adaptive
from skimage import segmentation
from skimage import measure
from imutils import perspective
import numpy as np
import imutils
import cv2


'''
collections.namedtuple: 產生一種新的 type, 可以用 "名稱" 來存取 element 的 sub-class of tuple
跟 C 的 struct 很像
這個新 type 的 名字為 LicensePlateRegion，而它的 attribute 是一個 list 包含了 4 個 attributes
'''
# define the named tupled to store the license plate
# plate     : 影像裡辨識為車牌的部份 (image-RGB)
# candidates: plate 裡辨識為字的部份的 全白遮罩
# thresh    : plate 的影像但移除了認為不是字的部份 (binary-image)
LicensePlate = namedtuple("LicensePlateRegion", ["success", "plate", "thresh", "candidates"])

class LicensePlateDetector:
    def __init__(self, image, minPlateW=60, minPlateH=20, numChars=7, minCharW=40, displayAll=False):
        self.image = image
        self.minPlateW = minPlateW
        self.minPlateH = minPlateH
        self.numChars = numChars
        self.minCharW = minCharW
        self.displayAll = displayAll

    def detect(self):
        # 1. detect license plate regions in the image
        lpRegions = self.detectPlates()

        # 2. loop over the license plate regions
        for lpRegion in lpRegions:
            # detect character candidates in the current license plate region
            lp = self.detectCharacterCandidates(lpRegion)

            # only continue if characters were successfully detected
            if lp.success:
                chars = self.scissor(lp)
                # yield a tuple of the license plate object and bounding box
                # yield 返回的是一個 generator
                yield(lpRegion, chars)


    # 取得 plate 上 所有的 contour
    def detectPlates(self):
        print("detectplates")
        regions = []

        # 因為車牌的寛度比高度長，所以將 kernel 設為 (13, 5)
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))

        # 用來產生 mask 時所需的 morphology operation 的 kernel size
        # squareKernel 配合 close operation 使用，用來關閉 image 裡的一些 gap
        squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # 使用 MORPH_BLACKHAT 來凸顯 black region (車牌的字) 跟 white region (車牌的背景) 對比
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
        if self.displayAll == True:
            cv2.imshow("blackhat", blackhat)
            cv2.waitKey(0)

        # light 是 image 中比較亮的部份，會被拿來當作 mask，主要是將一些小的 gap 閉起來 (closing)
        light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKernel)
        if self.displayAll == True:
            cv2.imshow("light morph close", light)
            cv2.waitKey(0)

        # 再將 close 後的 image 裡
        # grayscale pixel intensity > 50 的設為 255
        # 其它的設為 0
        light = cv2.threshold(light, 50, 255, cv2.THRESH_BINARY)[1]
        if self.displayAll == True:
            cv2.imshow("light threshold", light)
            cv2.waitKey(0)

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
            # if aspectRatio > 2.9:
            #     cv2.drawContours(self.image, [box], -1, (0, 255, 0), 2)
            #     cv2.putText(self.image, "%s" % (aspectRatio), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
                # print("ratio: {}, w:{}, h:{}".format(aspectRatio, w, h))
            # for test

            if (aspectRatio > 2.8 and aspectRatio < 6) and h > self.minPlateH and w > self.minPlateW:
                regions.append(box)
        # cv2.imshow("boxed", self.image)
        # cv2.waitKey(0)
        return regions

    # 取得 認為可能是"字" 的位置的 全白的 mask
    def detectCharacterCandidates(self, region):
        # 將 plate region 做俯視的變形
        plate = perspective.four_point_transform(self.image, region)
        # cv2.imshow("Perspective Transform", imutils.resize(plate, width=400))

        # 取出 HSV 裡的 Value(lightness)，因為 Value 表示 lightness
        # 用 Value 來做 threshold of dark or light 的效果會比用 grayscale 來得好
        V = cv2.split(cv2.cvtColor(plate, cv2.COLOR_BGR2HSV))[2]

        # 使用 skimage.filters 的 threshold_adaptive
        # 高於 threshold 條件的 pixel 變為白色 即 車牌背景
        # 低於 threshold 條件的 pixel 變為零(黑色) 即 字元
        # thresholding is applied to each local 29x29 pixel region
        thresh = threshold_adaptive(V, 29, offset=15).astype("uint8") * 255
        # 將 threshold 後的結果反向
        # 則 車牌背景變為 黑色
        # 而 字元變為 白色
        thresh = cv2.bitwise_not(thresh)

        # 縮圖到 width = 400
        plate = imutils.resize(plate, width=400)
        thresh = imutils.resize(thresh, width=400)
        cv2.imshow("Thresh", thresh)

        # see the explanation at the BOTTOM
        # scikit-image  <= 0.11.X, the background label was originally -1
        # scikit-image  >= 0.12.X, the background label is 0
        # labels = measure.label(thresh, neighbors=8, background=0)
        labels = measure.label(thresh, background=0)
        print("labels.shape: {}".format(labels.shape))
        # charCandidates 是一個 binary image
        charCandidates = np.zeros(thresh.shape, dtype="uint8")

        for label in np.unique(labels):
            # if this is the background label, ignore it
            if label == 0:
                continue

            labelMask = np.zeros(thresh.shape, dtype="uint8")
            labelMask[labels == label] = 255
            (cnts, _) = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(cnts) > 0:
                # grab the largest contour
                c = max(cnts, key=cv2.contourArea)
                # grab the bounding box for the contour
                (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)

                # compute the aspect ratio, solidity, and height ratio for the component
                aspectRatio = boxW / float(boxH)
                # solidity: contour 裡的 pixel 佔 convexHull 的比例
                solidity = cv2.contourArea(c) / float(boxW * boxH)
                # the ratio between component height and plate height
                heightRatio = boxH / float(plate.shape[0])

                # the rule of character
                # 這是經驗值，需視不同的車牌的字做調整
                keepAspectRatio = aspectRatio < 1.0
                keepSolidity = solidity > 0.15
                keepHeight = heightRatio > 0.4 and heightRatio < 0.95

                if keepAspectRatio and keepSolidity and keepHeight:
                    # 取得 convexHull (圍住所有 convex point(凸出的點) 的範圍)
                    # 而這 convexHull 就是我們的 character candidates
                    hull = cv2.convexHull(c)

                    # 將 convexHull 以內填滿白色的方式畫到 charCandidates
                    # 而 charCandidates 就是我們的 character candidates mask
                    cv2.drawContours(charCandidates, [hull], -1, 255, -1)

        # 清除任何 "碰到" character candidates mask 的 pixel
        charCandidates = segmentation.clear_border(charCandidates)

        (cnts, _) = cv2.findContours(charCandidates.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.imshow("Original Candidates", charCandidates)

        # 去掉認為不是 車牌的字的部份
        if len(cnts) > self.numChars:
            (charCandidates, cnts) = self.pruneCandidates(charCandidates, cnts)
            cv2.imshow("Pruned Candidates", charCandidates)

        thresh = cv2.bitwise_and(thresh, thresh, mask=charCandidates)
        cv2.imshow("Final Char Threshold", thresh)


        # 返回我們自訂的 "struct" LicensePlate
        return LicensePlate(success=True, plate=plate, thresh=thresh, candidates=charCandidates)


    def pruneCandidates(self, charCandidates, cnts):

        prunedCandidates = np.zeros(charCandidates.shape, dtype="uint8")
        dims = []

        mtarray = []
        avgY = 0
        avgHeight = 0
        rects = []
        # get bundingbox of contour
        for c in cnts:
            (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)
            rects.append((boxX, boxY, boxW, boxH))
            dims.append(boxY + (boxH * 0.5))
            avgY += (boxY + boxH)
            avgHeight += boxH


            # test
            # print("rects:{}".format(rects))
            print("y:{}, h:{}".format(boxY, boxH))
            # print("(boxY + boxH) * 0.5 = {}".format((boxY + boxH) * 0.5))
            print("#######    x: {}, y+h:{}".format(boxX, boxY + boxH))
            mtarray.append([boxX, boxY + boxH])
            # test

        # @MT
        avgY /= len(cnts)
        avgHeight /= len(cnts)
        print("avgY = {}".format(avgY))
        print("avgHeight = {}".format(avgHeight))

        mtarray = np.array(mtarray)
        # 以 mtarray 的 column 0 來排序，6gjvr.jpg 的第 7 個字母
        # 是由左往右數來的 第 9 個 blob, 以下面的方法會被濾掉
        # 下面的方法非常不準確，要再修改
        sortmt = mtarray[np.argsort(mtarray[:, 0]), :]
        print("sort:{}".format(sortmt))
        # @MT

        dims = np.array(dims)
        diffs = []
        selected = []

        # 目前看來這個方法會誤判 6gjvr.jpg 的數字 1 而把它去掉，
        for i in xrange(0, len(dims)):
            print("dims - dims[{}].sum: {}".format(i,np.absolute(dims - dims[i]).sum()))
            diffs.append(np.absolute(dims - dims[i]).sum())

        mtarray1 = np.argsort(diffs)
        print("mtarray1: {}".format(mtarray1))

        for index in np.argsort(diffs)[:self.numChars]:
        # count = 0
        # for index in np.argsort(diffs):

            # if rects[index][0] < 10:
            #     print("skip index = {}, x:{}".format(index, rects[index][0]))
            #     continue
            # if np.absolute(rects[index][0] - charCandidates.shape[1]) < 10:
            #     print("skip index = {}, x:{}".format(index, rects[index][0]))
            #     continue
            # count += 1
            # if count < 8:
            print("index = {}, x:{}".format(index, rects[index][0]))
            cv2.drawContours(prunedCandidates, [cnts[index]], -1, 255, -1)
            selected.append(cnts[i])

        return (prunedCandidates, selected)


    def scissor(self, lp):
        # 透過 candidate(全白遮罩) 取得 contour
        (cnts, _) = cv2.findContours(lp.candidates.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        chars = []

        for c in cnts:
            (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)

            # 這裡是想固定顯示車牌字時影像的寛度，還要再修改
            # dX = min(self.minCharW, self.minCharW - boxW) / 2
            # boxX -= dX
            # boxW += (dX * 2)
            # end

            boxes.append((boxX, boxY, boxX + boxW, boxY + boxH))

        # sort bounding box from left to right
        boxes = sorted(boxes, key=lambda b:b[0])

        for(startX, startY, endX, endY) in boxes:
            # extract content in bounding box of thresh(binary)
            chars.append(lp.thresh[startY:endY, startX:endX])

        return chars

    @staticmethod
    def preprocessChar(char):
        # extract ROI from the char()
        (cnts, _) = cv2.findContours(char.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cnts, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(c)
        char = char[y:y+h, x:x+w]

        return char







# ######################################################################################################

'''
skimage.measure.label(input, neighbors=None, background=None, return_num=False, connectivity=None)
input : ndarray of dtype int Image to label.
Parameters:

    neighbors : {4, 8}, int, optional
        Whether to use 4- or 8-“connectivity”. In 3D, 4-“connectivity” means connected pixels have to share face, whereas with 8-“connectivity”, they have to share only edge or vertex. Deprecated, use ``connectivity`` instead.

    background : int, optional
        Consider all pixels with this value as background pixels, and label them as 0. By default, 0-valued pixels are considered as background pixels.

    return_num : bool, optional
        Whether to return the number of assigned labels.

    connectivity : int, optional
        Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor. Accepted values are ranging from 1 to input.ndim. If None, a full connectivity of input.ndim is used.

Returns:
    labels : ndarray of dtype int
    Labeled array, where all connected regions are assigned the same integer value.

    num : int, optional
    Number of labels, which equals the maximum label index and is only returned if return_num is True.

for instance

a = array([[1, 0, 1],
       [0, 2, 0],
       [0, 1, 1]])

labels = measure.label(a)
       = array([[1, 0, 2],
               [0, 3, 0],
               [0, 4, 4]])

np.unique(labels) = array([0, 1, 2, 3, 4])


for label in np.unique(labels):
    if label == 0:
        continue
    mask = np.zeros(a.shape, dtype="uint8")
    mask[labels == label] = 255
    print(mask)

output :
    [[255   0   0]
     [  0   0   0]
     [  0   0   0]],
    [[  0   0 255]
     [  0   0   0]
     [  0   0   0]],
    [[  0   0   0]
     [  0 255   0]
     [  0   0   0]],
    [[  0   0   0]
     [  0   0   0]
     [  0 255 255]]


'''

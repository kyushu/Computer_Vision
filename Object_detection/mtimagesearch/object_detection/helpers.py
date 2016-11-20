
#-*- coding: utf-8 -*-　　
#-*- coding: cp950 -*-　
# 要使用中文就要加上面兩行

'''
Purpose:
    Helper Class for pyramid process and crop ROI from image

openCV 使用 pyrUp 跟 pyDown 來做 pyramid，每個 pyramid level 都會做 Gaussian blurring smooth
的動作，不過這樣會嚴重影響效能，所以我的 pyramid 不做這 Gaussian blurring smooth 這個動作
而且大部份的時候在 pyramid 的每個 layer 做 Gaussian blurring smooth 會降底準確度
(accuracy drops when smoothing is performed at each layer)
'''

import imutils
import cv2

# 使用 yield 的 function 稱為 generator
# 所謂的 genrator 是該 function 每次執行會記住這次執行完的結果
# 並回傳符合可迭代者與迭代器介面的物件，而就是 generator object
#
# 所以若是 result = pyramid(image)，則 result 是 generator object 而不是跟 image 一樣是一個 Array
# 要取得從 result 取得 pyramid(image) resized 後的 image，要使用 next(result)
# next(result) = 這一次執行 pyramid 的 resized image

# 也可以使用 generator expression 來取得 generator 的結果
# (算式 for 名稱 in 可迭代者)
# (算式 for 名稱 in 可迭代者 if 運算式)
# 例如：
# for layer in pyramid(image):
    # layer = next(pyramid(image)) = resized image

def pyramid(image, scale=1.5, minSize=(30,30)):
    # yield the original image
    # 有可能 image 的 size 已經小於 minSize 了
    # 所以一進 while 就 break，而這時要回傳的是原本傳進來的 image
    yield image

    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        yield image



# xrange 是 for 產生器運算式 (generator expression) 使用的，也就是每次取得一個 generator 的內容
# 一次只返回 generator 的一個內容
# xrange(start, end, step)

# range 是 for 串列生成式 (list comprehension) 所使用的，也就是取得 list 的內容
# 一次返回一個 list 的內容
def sliding_window(image, stepSize, windowSize):
    # slide a window across the iamge
    for y in xrange(0, image.shape[0], stepSize):
        for x in xrange(0, image.shape[1], stepSize):
            # yield the current window
            yield(x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


# Crop the ROI to a fixed size (dstSize)
def crop_ct101_bb(image, bb, padding=10, dstSize=(32, 32)):

    # get bounding box of ROI
    (y, h, x, w) = bb

    # 設邊界值
    (x, y) = (max(x - padding, 0), max(y - padding, 0))

    # Crop ROI from image by it's Bounding box (add Padding range)
    # adding a bit padding can actually increase the accuracy of our HOG detector
    roi = image[y:h + padding, x:w + padding]

    # Resize ROI to dstSize
    roi = cv2.resize(roi, dstSize, interpolation=cv2.INTER_AREA)

    return roi

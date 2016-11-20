# -*- coding:utf-8 -*-

import helpers

class ObjectDetector:
    def __init__(self, model, desc):
        # store the classifier mode and HOG descriptor
        self.model = model
        self.desc = desc

    def detect(self, image, winDim, winStep=4, pyramidScale=1.5, minProb=0.7):
        boxes = []
        probs = []

        # loop over the image pyramid (scale down image by pyramidScale)
        for layer in helpers.pyramid(image, scale=pyramidScale, minSize=winDim):
            scale = image.shape[0] / float(layer.shape[0])

            # loop over the sliding widnows for hte current pyramid lyaer
            # 由左至右，以 sliding_window 的大小從目前的 pyramid layer 擷取影像內容
            for(x, y, window) in helpers.sliding_window(layer, winStep, winDim):
                # Get window dimensions
                (winH, winW) = window.shape[:2]

                # ensure the window dimensions match the supplied sliding window dimensions
                if winH == winDim[1] and winW == winDim[0]:
                    # Get the HOG features and reshape it to one raw array (features vector)
                    features = self.desc.describe(window).reshape(1, -1)
                    # use our trained model to predict by using this feature vector
                    prob = self.model.predict_proba(features)[0][1]

                    # if probability over the min threshold, then add it into the result array
                    if prob > minProb:
                        (startX, startY) = (int(scale * x), int(scale * y))
                        endX = int(startX + (scale * winW))
                        endY = int(startY + (scale * winH))

                        boxes.append((startX, startY, endX, endY))
                        probs.append(prob)

        return (boxes, probs)




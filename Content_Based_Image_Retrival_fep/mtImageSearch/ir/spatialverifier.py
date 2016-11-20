# -*- coding:utf-8 -*-

'''
RANSAC: Random Sample Consensus
'''


from searchresult import SearchResult
from sklearn.metrics import pairwise
import numpy as np
import datetime
import h5py
import cv2
class SpatialVerifier:
    def __init__(self, featuresDBPath, idf, vocab, ratio=0.75, minMatches=10, reprojThresh=4.0):
        # idf : The inverse document frequency counts
        self.idf = idf
        # vocab : The visual vocabulary
        self.vocab = vocab
        # ration : David Lowe’s raw feature matching ratio, useful for false-positive match pruning
        self.ratio = ratio
        # minMatches : minimum number of matcher required to perform spatial verification
        self.minMatches = minMatches
        # reprojThresh : threshold for computing inlier points (The re-projection threshold for the RANSAC algorithm)
        self.reprojThresh = reprojThresh

        # Load the feature dataset
        self.featuresDB = h5py.File(featuresDBPath)

    # queryKps      : The keypoints detected in the query image
    # queryDescs    : The local invariant descriptors extracted from the query image
    # searchResult  : object returned from the search  method of Searcher
    # numResults    :
    def rerank(self, queryKps, queryDescs, searchResult, numResults=10):
        startTime = datetime.datetime.now()
        reranked = {}

        resultIdxs = np.array([r[-1] for r in searchResult.results])
        resultIdxs.sort()

        for (i, (start, end)) in zip(resultIdxs, self.featuresDB["index"][resultIdxs, ...]):

            # grab the features from featruesDB which is
            # [keypointX, keypointY, description_1, description_2, ..., description_N]
            rows = self.featuresDB["features"][start: end]
            (kps, descs) = (rows[:, :2], rows[:, 2:])

            # determine matched inlier keypoint and grab the indexes of the matched keypoints
            bovwIdxs = self.match(queryKps, queryDescs.astype("float32"), kps, descs.astype("float32"))

            # sum the socre of the idf value
            if bovwIdxs is not None:
                score = self.idf[bovwIdxs].sum()
                reranked[i] = score

        if len(reranked) == 0:
            return searchResult

        # test
        # mttest = [mtv, self.featuresDB["image_ids"][0]]
        # print("mttest: {}".format(mttest))
        # test
        results = sorted( [(v, self.featuresDB["image_ids"][k], k) for (k, v) in reranked.items()], reverse=True )

        for (score, imageID, imageIdx) in searchResult.results:
            if imageIdx not in reranked:
                results.append((score, imageID, imageIdx))

        return SearchResult(results[:numResults], (datetime.datetime.now() - startTime).total_seconds())


    def match(self, kpsA, featuresA, kpsB, featuresB):
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresB, featuresA, 2)
        matches = []
        inlierIdxs = None

        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * self.ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        if len(matches) >= self.minMatches:
            ptsA = np.float32([kpsA[i] for (i, _) in matches])
            ptsB = np.float32([kpsB[j] for (_, j) in matches])

            (_, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, self.reprojThresh)

            idxs = np.where(status.flatten() > 0)[0]
            inlierIdxs = pairwise.euclidean_distances(featuresB[idxs], Y=self.vocab)
            inlierIdxs = inlierIdxs.argmin(axis=1)

        return inlierIdxs

    def finish(self):
        # close the features database
        self.featuresDB.close()

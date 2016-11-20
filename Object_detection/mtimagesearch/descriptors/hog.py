# -*- coding:utf-8 -*-

'''
Purpose:
    encapsulate sklearn.feature.hog
'''

from skimage import feature


class HOG:

	# initialize with necessary parameters
    def __init__(self, orientations=12, pixelsPerCell=(4, 4), cellsPerBlock=(2, 2), normalize=True):

        self.orientations = orientations
        self.pixelsPerCell = pixelsPerCell
        self.cellsPerBlock = cellsPerBlock
        self.normalize = normalize


    def describe(self, image):
        '''
        image : (M, N)
            ndarray Input image (greyscale).

        orientations : int
            Number of orientation bins.

        pixels_per_cell : 2 tuple (int, int)
            Size (in pixels) of a cell.

        cells_per_block : 2 tuple (int,int)
            Number of cells in each block.

        visualise : bool, optional
            Also return an image of the HOG.

        transform_sqrt : bool, optional
            Apply power law compression to normalise the image before processing. DO NOT use this if the image contains negative values. Also see notes section below.

        feature_vector : bool, optional
            Return the data as a feature vector by calling .ravel() on the result just before returning.

        normalise : bool, deprecated
            The parameter is deprecated. Use transform_sqrt for power law compression. normalise has been deprecated.

        Returns:
            newarr : ndarray
                HOG for the image as a 1D (flattened) array.
            hog_image : ndarray (if visualise=True)
                A visualisation of the HOG image.
        '''
        hist = feature.hog(image, orientations=self.orientations, pixels_per_cell=self.pixelsPerCell,
                           cells_per_block=self.cellsPerBlock, transform_sqrt=self.normalize)
        hist[hist < 0] = 0

        return hist

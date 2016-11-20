# -*- coding:utf-8 -*-

'''
usage:
    python extract_pbow.py --features-db output/training_features.hdf5 --codebook output/vocab.cpickle --levels 2 --pbow-db output/training_pbow.hdf5
'''

# import the necessary packages
from mtImageSearch.ir import BagOfVisualWords
from mtImageSearch.indexer import BOVWIndexer
from mtImageSearch.descriptors import PBOW
import argparse
import cPickle
import h5py

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--features-db", required=True, help="Path the features database")
ap.add_argument("-c", "--codebook", required=True, help="Path to the codebook")
ap.add_argument("-b", "--pbow-db", required=True, help="Path to where the pyramid of bag-of-visual-words database will be stored")
ap.add_argument("-s", "--max-buffer-size", type=int, default=100, help="Maximum buffer size for # of features to be stored in memory")
ap.add_argument("-l", "--levels", type=int, default=2, help="# of pyramid levels to generate")
args = vars(ap.parse_args())

# load the codebook vocabulary
vocab = cPickle.loads(open(args["codebook"]).read())
# initialize bovw with vocab
bovw = BagOfVisualWords(vocab)
# initialize pbow with bovw and numLevels parameter
pbow = PBOW(bovw, numLevels=args["levels"])


featureDim = PBOW.featureDim(bovw.codebook.shape[0], args["levels"])
featuresDB = h5py.File(args["features_db"], mode="r")
bi = BOVWIndexer(featureDim, args["pbow_db"], estNumImages=featuresDB["image_ids"].shape[0],
    maxBufferSize=args["max_buffer_size"])


# loop over the image IDs
for (i, imageID) in enumerate(featuresDB["image_ids"]):
    # grab the image dimensions, along with the index lookup values from the
    # database
    (h, w) = featuresDB["image_dims"][i]
    (start, end) = featuresDB["index"][i]

    # check to see if progress should be displayed
    if i > 0 and i % 10 == 0:
        bi._debug("processed {} images".format(i), msgType="[PROGRESS]")

    # extract the keypoints and feature vectors for the current image using the
    # starting and ending offsets (while ignoring the keypoints) and then create
    # the pyramid of bag-of-visual-words representation
    kps = featuresDB["features"][start:end][:, :2]
    descs = featuresDB["features"][start:end][:, 2:]
    hist = pbow.describe(w, h, kps, descs)

    # add the bag-of-visual-words to the index
    bi.add(hist)

# close the features database and finish the indexing process
featuresDB.close()
bi.finish()

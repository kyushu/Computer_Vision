import numpy as np

# calculate chi-squared distance
def chi2_distance(histA, histB, eps=1e-10):
    d = 0.5 * np.sum( ((histA - histB)**2) / (histA + histB + eps) )

    return d

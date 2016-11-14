# Typical/standard machine learning algorithm implementations
# Generalized to be independent of dimension
from __future__ import print_function, division
import numpy as np
import scipy.spatial.distance as spd
from math import sqrt

# Performs K-Nearest Neighbors on the given data and returns the k neighbors
# c: data to perform KNN on
# dataSet: The "labeled" data
# k: number of neighbors to compute
# dist_eq: What kind of distance equation to use in the computation
# p: Value of "p" used in Minkowski Distance Calculation
# r_dist: True or False to return a list of the closest distances
def KNN(c, dataSet, k=3, dist_eq="Euclidean", p=3, r_dist=False):
    neighbors = []
    dist = []
    for data in dataSet:
        distance = 0
        if dist_eq == "Euclidean":
            distance = spd.euclidean(c, data)
        elif dist_eq == "Manhattan":
            distance = spd.cityblock(c, data)
        elif dist_eq == "Minkowski":
            distance = spd.minkowski(c, data, p)
        elif dist_eq == "Hamming":
            distance = spd.hamming(c, data)
        else:
            raise ValueError("Invalid setting for dist_eq=Euclidean, Manhattan, Minkowski, Hamming")
        if len(neighbors) < k:
            neighbors.append(data)
            dist.append(distance)
        else:
            max_d = max(dist)
            if distance < max_d:
                indx = dist.index(max_d)
                neighbors[indx] = data
                dist[indx] = distance
    if r_dist == False:
        return neighbors
    else:
        return neighbors, dist

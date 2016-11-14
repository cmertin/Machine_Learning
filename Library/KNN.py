from __future__ import print_function, division
import numpy as np
import scipy.spatial.distance as spd
from math import sqrt

def KNN(c, dataSet, k=3, dist_eq="Euclidean", p=3, r_dist=False):
    neighbors = []
    dist = []
    index = 0
    for data in dataSet:
        #print("idx", index)
        #print("\n\n")
        #print(c)
        #print("\n")
        #print(data)
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
        index = index + 1
    if r_dist == False:
        return neighbors
    else:
        return neighbors, dist

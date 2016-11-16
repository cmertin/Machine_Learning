# Typical/standard machine learning algorithm implementations
# Generalized to be independent of dimension
from __future__ import print_function, division
import numpy as np
import numpy.linalg as npl
import scipy.spatial.distance as spd
from math import sqrt
'''
# y is the numebr of data points
def J_derivative(theta, x, y, h):
    temp = 0
    for i in range(0, len(y)):
        temp = temp + (h(x[i], theta) - y[i]) * x[i])
    return temp/len(y)
# Computes the gradient descent to minimize the function and get the best fitting
# values for theta_0 and theta_1. It continually loops until the frobenius norm
# of the previous values of theta minus the new values of theta is less than some
# tolerance (defaults to 1e-6).
# alpha is the learning rate
# This function stores each value of theta over each iteration so that the
# evolution of theta can be plotted. It returns the final value as "theta" and
# it returns a list of values as theta_data
def GradientDescent(theta, x, y, alpha, tol=1e-6, h=h_linear):
    theta_old = copy.deepcopy(theta) + 10
        while npl.norm(theta - theta_old) > tol:
            theta_data.append(theta.copy())
            theta_old = theta.copy()
            d_dt = J_Derivative(theta, x, y)
            theta[0] = theta[0] - alpha * d_dt[0]
            theta[1] = theta[1] - alpha * d_dt[1]
    return theta, theta_data


def Linear_Regression(x, y, alpha, tol=1e-6, h=h_linear):
    theta = np.zeros(len(y))
'''

# The values are simply theta^T * x
# x: Feature vector
# theta: Fit theta parameters
def CalculateResult(x, theta):
    return np.dot(np.transpose(theta), x)

# Returns the values of theta based on the Normal Equation. This is a direct
# solve for the minimum values of theta wihtout having to iterate with
# gradient descent, as most of the other interpolating functions do. Instead it
# calculates the results by using the analytical expression
# Note: Not for large matrices, ie bigger than 10k x 10k
# X: Matrix with each row being a feature vector
# y: 1D vector where each index is the "true value"
def NormalEquation(X, y):
    XT = np.transpose(X)    # X^T
    temp = np.dot(XT, X)    # X^T * X
    temp = npl.pinv(temp)   # Calcualtes the pseudo-inverse to deal with singularity
    temp = np.dot(temp, XT) # (X^T * X)^-1 * X^T
    return np.dot(temp, y)  # theta = (X^T * X)^-1 * X^T * y


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

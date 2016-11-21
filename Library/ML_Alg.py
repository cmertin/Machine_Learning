# Typical/standard machine learning algorithm implementations
# Generalized to be independent of dimension
from __future__ import print_function, division
import numpy as np
import numpy.linalg as npl
import scipy.spatial.distance as spd
from sklearn.utils import shuffle
import copy as copy
from math import sqrt

def UpSample(x, y, test=.2):
    test_size = int(len(x) * test)
    train_size = len(x) - test_size
    positive_x = []
    positive_y = []
    negative_x = []
    negative_y = []
    test_x = []
    test_y = []
    x_data = []
    y_data = []

    # Separates positives and negatives
    for i in range(len(y)-1, -1, -1):
        if y[i] == 1:
            positive_x.append(x[i])
            positive_y.append(y[i])
        else:
            negative_x.append(x[i])
            negative_y.append(y[i])

    # Shuffles the data
    p_x, p_y = shuffle(positive_x, positive_y, random_state=0)
    n_x, n_y = shuffle(positive_x, positive_y, random_state=0)

    ratio = len(p_y)/len(n_y)
    print(ratio)


# Determines the accuracy for a given set of parameters theta, and returns
# the percentage of those that it got right versus the entire data set
def Accuracy(theta, test_x, test_y, h):
    true_p = 0
    true_n = 0
    false_p = 0
    false_n = 0
    for i in range(0, test_x.shape[0]):
        y = h(theta, test_x)
        if y == 0:
            if test_y[i] == 0:
                true_n = true_n + 1
            else:
                false_n = false_n + 1
        elif y == 1:
            if test_y[i] == 1:
                true_p = true_p + 1
            else:
                false_p = false_p + 1
    return (true_p + true_n)/test_x.shape[0]

# Hypothesis function for linear regression
def h_linear(x, theta):
    return np.inner(x, theta)

# Hypothesis function for logistic regression
def h_logistic_regression(x, theta):
    exponent = np.inner(x, theta)
    denominator = 1 + np.exp(-exponent)
    return 1/denominator

# Calcualtes the d/dt of the cost function (J) given a hypothesis
# function. Used for Gradient Descent. It returns a vector of d/dt
# for each feature
def J_Derivative(theta, x, y, h):
    d_dt = []
    for feature in range(0, x.shape[1]):
        temp = 0
        for data in range(0, x.shape[0]):
            temp = temp + (h(x[data], theta) - y[data]) * x[data][feature]
        d_dt.append(temp / x.shape[0])
    return d_dt

# Method of Gradient Descent, to find the minimum for the classification function.
# theta: Inital guess for the parameters
# x: Data points size: [N,M] (N data points, M features per data)
# y: The "true/labeled" values for the given data
# alpha: The learning rate for the function
# lambda_: Used for regularization. Using lambda = 0 ignores regularization
# tol: The tolerance/stopping point for the vector
# updates: If true, it returns a vector theta_updates which contains the values
#          of theta for each iteration so you can see how it evolves
def GradientDescent(theta, x, y, alpha, h, lambda_=0, tol=1e-6, updates=False):
    theta_old = copy.deepcopy(theta) + 10
    theta_updates = []
    while npl.norm(theta - theta_old) > tol:
        if updates == True:
            theta_updates.append(theta.copy())
        theta_old = theta.copy()
        d_dt = J_Derivative(theta, x, y, h)
        for i in range(0, theta.shape[0]):
            theta[i] = theta[i] * (1 - alpha * lambda_ / len(x)) - alpha * d_dt[i]
    if updates == True:
        return theta, theta_updates
    else:
        return theta

# Performs Linear Regression on given data: See GradientDescent for arguments
def LinearRegression(theta, x, y, alpha, lambda_=0, tol=1e-6, updates=False):
    return GradientDescent(theta, x, y, alpha, h_linear, lambda_, tol, updates)

# Performs Logistic Regression on given data: See GradientDescent for arguments
def LogisticRegression(theta, x, y, alpha, lambda_=0, tol=1e-6, updates=False):
    return GradientDescent(theta, x, y, alpha, h_logistic_regression, lambda_, tol, updates)

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

# Normal Equation with Regularization
def NormalEquation_Reg(X, y, lambda_):
    XT = np.transpose(X)    # X^T
    l_mat = lambda_ * np.eye(XT.shape[0]) # lambda along the diagonal
    l_mat[0][0] = 0         # Except the first term
    temp = np.dot(XT, X)    # X^T * X
    temp = temp + l_mat     # X^T * X + lambda * I
    temp = npl.pinv(temp)   # Calcualtes the pseudo-inverse to deal with singularity
    temp = np.dot(temp, XT) # (X^T * X + lambda * I)^-1 * X^T
    return np.dot(temp, y)  # theta = (X^T * X + lambda * I)^-1 * X^T * y

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

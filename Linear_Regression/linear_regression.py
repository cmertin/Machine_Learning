from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npl

# Reads in a file line by line and returns a vector of the values
# as a numpy array
def ReadData(filename):
    data = []
    lines = [line.rstrip('\n') for line in open(filename)]
    for line in lines:
        data.append(float(line))
    return np.asarray(data)

# Returns the value of the linear function based on the values of
# theta_0, theta_1, and x
def h(theta0, theta1, x):
    return theta0 + theta1 * x

# Returns the values of the derivative of J for *both* d/dt_0 and d/dt_1
# as a list so they don't have to be computed independently and helps
# in making sure a _simultaneous update_ is performed
def J_Derivative(theta, x, y):
    derivative = []
    t0 = theta[0]
    t1 = theta[1]
    dt0 = 0
    dt1 = 0
    for i in range(0, len(x)):
        dt0 = dt0 + (h(t0, t1, x[i]) - y[i])
        dt1 = dt0 + (h(t0, t1, x[i]) - y[i]) * x[i]
    dt0 = dt0/len(x)
    dt1 = dt1/len(x)
    return [dt0, dt1]

# Computes the gradient descent to minimize the function and get the best fitting
# values for theta_0 and theta_1. It continually loops until the frobenius norm
# of the previous values of theta minus the new values of theta is less than some
# tolerance (defaults to 1e-6).
# alpha is the learning rate
# This function stores each value of theta over each iteration so that the
# evolution of theta can be plotted. It returns the final value as "theta" and
# it returns a list of values as theta_data
def GradientDescent(theta, x, y, alpha, tol=1e-6):
    theta_old = theta.copy() + 10
    theta_data = []
    while npl.norm(theta - theta_old) > tol:
        theta_data.append(theta.copy())
        theta_old = theta.copy()
        d_dt = J_Derivative(theta, x, y)
        theta[0] = theta[0] - alpha * d_dt[0]
        theta[1] = theta[1] - alpha * d_dt[1]
    return theta, theta_data

# This function takes in some given values of x and evaluates the hypothesis
# function h(x) at all of those values (with theta) so you can get a corresponding
# line to plot
def Eval_X(theta, x):
    vals = []
    for i in range(0, len(x)):
        temp = h(theta[0], theta[1], x[i])
        vals.append(temp)
    return vals

x_file = "age.dat"
y_file = "height.dat"
outfile = "linreg_1.png"
Plot_All = False

x = ReadData(x_file)
y = ReadData(y_file)
x_ = np.arange(0, 9.01, .01)
theta = np.zeros(2)
alpha = 0.25

plt.scatter(x, y)
plt.xlabel("Ages in Years")
plt.ylabel("Height in Meters")
plt.savefig(outfile, type="png", bbox_inches="tight")

theta, theta_data = GradientDescent(theta, x, y, alpha)

print("Final Vector: ")
print(theta)

if Plot_All == True:
    file_count = 0
    for sol in theta_data:
        vals = Eval_X(sol, x_)
        buffer = ""
        if file_count < 10:
            buffer = "00"
        elif file_count < 100 and file_count >= 10:
            buffer = "0"
        outfile = "Plot_" + buffer + str(file_count) + ".png"
        x0 = "%.5f" % sol[0]
        x1 = "%.5f" % sol[1]
        title = "y = " + x1 + "x + " + x0
        plt.clf()
        plt.scatter(x, y)
        plt.plot(x_, vals, 'r')
        plt.xlabel("Ages in Years")
        plt.ylabel("Height in Meters")
        plt.title(title)
        plt.xlim([1, 9])
        plt.ylim([0.7, 1.4])
        plt.savefig(outfile, type="png", bbox_inches="tight")
        file_count = file_count + 1
else:
    outfile = "plot_final.png"
    x0 = "%.5f" % theta[0]
    x1 = "%.5f" % theta[1]
    title = "y = " + x1 + "x + " + x0
    vals = Eval_X(theta, x_)
    plt.clf()
    plt.scatter(x, y)
    plt.plot(x_, vals, 'r')
    plt.title(title)
    plt.xlabel("Ages in Years")
    plt.ylabel("Height in Meters")
    plt.xlim([1, 9])
    plt.ylim([0.7, 1.4])
    plt.savefig(outfile, type="png", bbox_inches="tight")

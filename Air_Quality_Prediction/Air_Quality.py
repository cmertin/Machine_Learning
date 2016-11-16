from __future__ import print_function, division
import datetime
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(color_codes=True)
import pandas as pd
from scipy import stats, integrate
# Gets the path/library with my Machine Learning Programs and adds it to the
# current PATH so it can be imported
LIB_PATH = os.path.dirname(os.getcwd()) # Goes one parent directory up
LIB_PATH = LIB_PATH + "/Library/" # Appends the Library folder to the path
sys.path.append(LIB_PATH)
from ML_Alg import NormalEquation, CalculateResult

def ReadFile(filename):
    data = []
    lines = [line.rstrip('\n') for line in open(filename)]

    for line in lines:
        local = [1] # First element of x = 0, offset for theta
        temp = line.split(',')
        for item in temp:
            local.append(float(item.replace("\"","")))
        data.append(local)
    return data

BASE_DIR = os.getcwd()
inFile = BASE_DIR + "/Data/AirQuality_clean.csv"
error_dist = "Error_Distribution.pdf"
error_dist_png = "Error_Distribution.png"

data = ReadFile(inFile)

test_set = data[len(data)-24:]
data = data[:len(data)-24]

test_y = []
y_mat = []
thetas = []

# Pull out the y-values from the matrix and stores them in y_mat
for row in data:
    y_mat.append(np.asarray(row[4:14]))
    del row[5:15]

for row in test_set:
    test_y.append(np.asarray(row[4:14]))
    del row[5:15]

# Convert data to a matrix with each row being a feature vector
data = np.asarray(data)
test_set = np.asarray(test_set)
# Convert y_mat so that each row represents a value of y for each feature vector
y_mat = np.asarray(y_mat)
y_mat = np.transpose(y_mat)
test_y = np.asarray(test_y)

# Calculate the values of theta for each of the features in y
for y in y_mat:
    theta = NormalEquation(data, y)
    thetas.append(theta)

hours = []
result_mat = np.zeros([len(test_set), len(thetas)])
rel_err_mat = np.zeros([len(test_set), len(thetas)])




pollutant = []
pol_labels = ["CO", "Tin Oxide (PT08.S1)", "Non-Metanic HydroCarbons", "Benzene", "Titania (PT08.S2)", "NOx", "Tungsten Oxide (PT08.S3)", "NO2", "Tungsten Oxide (PT08.S4)", "Indium Oxide (PT08.S5)"]
# Calculate the approximate results from the thetas and the relative error
for i in range(0, len(thetas)):
    for j in range(0, len(test_set)):
        result = CalculateResult(test_set[j], thetas[i])
        denominator = max(abs(result), abs(test_y[j][i]))
        if test_y[j][i] == 0:
            rel_err = abs(result)
        else:
            rel_err = abs(result - test_y[j][i])/denominator
        pollutant.append([j, result, test_y[j][i], rel_err, pol_labels[i]])

# Puts the values in a "pandas.dataFrame" for seaborn plotting
df = pd.DataFrame(pollutant, columns=["Hours", "Concentration", "True Concen.", "Relative Error", "Particulate"])


# Plots the distribution of the relative errors
# Setup matplotlib.pyplot figure
f, axes = plt.subplots(5, 2, sharex=True, sharey=False)

init = 0
for i in range(0, 5):
    for j in range(0, 2):
        one = init
        two = one + 24
        s = df[one:two]
        sns.distplot(s["Relative Error"], kde=False, fit=stats.gamma, ax=axes[i, j])
        if i == 0 and j == 0:
            axes[i,j].set_ylim(0, 5)
        if i == 1 and j == 0:
            axes[i,j].set_ylim(0, 10)
        if i == 4 and j == 0:
            axes[i,j].set_ylim(0, 4)
        if i == 4 and j == 1:
            axes[i,j].set_ylim(0, 10)
        if i == 3 and j == 1:
            axes[i,j].set_ylim(0, 6)
        if i == 4:
            axes[i,j].set_xlabel("Relative Error")
        else:
            axes[i,j].set_xlabel("")
        axes[i,j].set_ylabel("Count")
        axes[i,j].set_xlim(-.1,1.05)
        axes[i,j].set_title(pol_labels[i * 2 + j])
        init = two

figure = plt.gcf() # get current figure
figure.set_size_inches(8, 8)
sns.plt.savefig("Error_Distribution.png", dpi=200, bbox_inches="tight")
#sns.plt.show()

# Plots the relative error for the hours predicted
lm = sns.lmplot(x="Hours", y="Relative Error", col="Particulate", hue="Particulate", data=df, col_wrap=3, size=4, sharey=False)
figure = plt.gcf() # get current figure
figure.set_size_inches(10, 8)
axes = lm.axes
for i in range(0, len(axes)):
    axes[i].set_xlim(-1, 25)
sns.plt.savefig("Relative_Error_Hour.png", dpi=200, bbox_inches="tight")
#sns.plt.show()

# Plots the predicted concentration per hour
mn_lst = [-3,0,900,100,0,700,200,600,50,950]
mx_lst = [25,5,1200,200,15,1000,400,900,200,1250]
lm = sns.lmplot(x="Hours", y="Concentration", col="Particulate", hue="Particulate", data=df, col_wrap=3, size=4, sharey=False)
figure = plt.gcf() # get current figure
figure.set_size_inches(10, 8)
axes = lm.axes
for i in range(0, len(axes)):
    axes[i].set_xlim(-1,25)
    axes[i].set_ylim(mn_lst[i], mx_lst[i])
sns.plt.savefig("Concentration_Hour.png", dpi=200, bbox_inches="tight")
#sns.plt.show()

# Plots the actual particulate concentration for the last 24 hours
mn_lst = [-3 ,0 ,700 ,0  ,0 ,400 ,0  ,300 ,0  ,600]
mx_lst = [25 ,5 ,1600,700,30,1500,700,1200,200,2000]
lm = sns.lmplot(x="Hours", y="True Concen.", col="Particulate", hue="Particulate", data=df, col_wrap=3, size=4, sharey=False)
figure = plt.gcf() # get current figure
figure.set_size_inches(10, 8)
axes = lm.axes
for i in range(0, len(axes)):
    axes[i].set_xlim(-1,25)
    axes[i].set_ylim(mn_lst[i], mx_lst[i])
sns.plt.savefig("True_Concentration_Hour.png", dpi=200, bbox_inches="tight")
#sns.plt.show()

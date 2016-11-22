from __future__ import print_function, division
import datetime
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(color_codes=True)
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from scipy import stats, integrate
# Gets the path/library with my Machine Learning Programs and adds it to the
# current PATH so it can be imported
LIB_PATH = os.path.dirname(os.getcwd()) # Goes one parent directory up
LIB_PATH = LIB_PATH + "/Library/" # Appends the Library folder to the path
sys.path.append(LIB_PATH)
from ML_Alg import LogisticRegression, UpSample
from f_io import ReadCSV

data_file = "creditcard.csv"
DIR = os.getcwd() + "/data/"
FILE = DIR + data_file

x, y = ReadCSV(FILE)

data, test_data = UpSample(x, y)

print("Finished Up Sampling Data")


svm = LinearSVC()
print("\nLinear SVM")
print("==========")
svm.fit(data[0], data[1])
y_pred = svm.predict(test_data[0])
print(classification_report(y_pred, test_data[1]))

log_reg = LogisticRegressionCV()
print("\nLogistic Regression")
print("===================")
log_reg.fit(data[0], data[1])
y_pred = log_reg.predict(test_data[0])
print(classification_report(y_pred, test_data[1]))


gnb = GaussianNB()
print("\nGaussian Naive Bayes")
print("====================")
gnb.fit(data[0], data[1])
y_pred = gnb.predict(test_data[0])
print(classification_report(y_pred, test_data[1]))

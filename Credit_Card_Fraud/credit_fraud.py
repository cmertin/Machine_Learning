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
from ML_Alg import LogisticRegression, UpSample
from f_io import ReadCSV

data_file = "creditcard.csv"
FILE = os.getcwd() + "/data/" + data_file

x, y = ReadCSV(FILE)

UpSample(x, y)

from __future__ import print_function, division
import datetime
import os
import sys
import numpy as np
# Gets the path/library with my Machine Learning Programs and adds it to the
# current PATH so it can be imported
LIB_PATH = os.path.dirname(os.getcwd()) # Goes one parent directory up
LIB_PATH = os.path.dirname(LIB_PATH) # Another parent directory up
LIB_PATH = LIB_PATH + "/Library/" # Appends the Library folder to the path
sys.path.append(LIB_PATH)
from KNN import *


# Reads in the csv file. It accepts a "blank string" such that it will search
# for the index that has missing data for a given row. If it a row has more than
# num_blank missing values, it throws the data away. This will allow you to choose
# how much "accuracy" you want to give to your data. To read in all of the data,
# set num_blank equal to -1.
def ReadFile(filename, miss_tkn="", num_blank=2, delim=','):
    assert num_blank >= -1, "num_blank must be >= -1"
    data = []
    lines = [line.rstrip('\n') for line in open(filename)]
    lines = lines[1:]
    for line in lines:
        curr_line = []
        ln = line.split(delim)
        if num_blank != -1:
            blank_count = ln.count(miss_tkn)
            if blank_count > num_blank:
                continue
        for i in range(0, len(ln)-2):
            if i == 0:
                curr_line.append(ln[0])
            elif i == 1:
                temp = ln[i].split('.')
                curr_line.append(int(temp[0]))
            else:
                temp = ln[i].replace(',','.')
                curr_line.append(float(temp))
        data.append(curr_line)
    return data

def KNN_Regression(dataSet, miss_tkn="", k=3, dist_eq="Euclidean", p=3):
    final_data = []
    none_lst = []
    none_idx = []
    missing_lst = []
    missing_idx = []
    idx = 0

    # Separate the data between missing and non-missing
    # Also stores the indices of each in the original list so that the
    # ordered list can be returned
    for data in dataSet:
        count = data.count(miss_tkn)
        if count == 0:
            none_lst.append(data[:])
            none_idx.append(idx)
        else:
            missing_lst.append(data[:])
            missing_idx.append(idx)
        idx = idx + 1

    # Perform KNN Regression and update the missing indices
    for i in range(0, len(missing_lst)):
        none_lst_rm = none_lst[:]
        miss_rm = missing_lst[i][:]

        # Gets all the indices in miss_rm that have the miss_tkn
        indx = [j for j, x in enumerate(miss_rm) if x == miss_tkn]

        # Removes the missing features from the lists to perform a more accurate
        # KNN Regression
        for j in range(len(indx)-1, -1, -1):
            idx = indx[j]
            miss_rm.pop(idx)
            for k_ in range(0, len(none_lst_rm)):
                if k_ == 725:
                    print(len(none_lst_rm[k_]), end="")
                none_lst_rm[k_].pop(idx)
                if k_ == 725:
                    print('\t', len(none_lst_rm[k_]))
        print(len(miss_rm))

        # Gets the k nearest neighbors now that value is removed
        k_n = KNN(miss_rm, none_lst_rm, k, dist_eq, p)

        # Find indices of the nearest neighbors in the original list with none
        # missing that has all the features
        match_idx = []
        for knn in k_n:
            for j in range(0, len(none_lst)):
                match_count = 0
                for k_ in range(0, 4):
                    if knn[k_] == none_lst[j][k_]:
                        match_count = match_count + 1
                if match_count == 4:
                    match_idx.append(j)
                    break

        # Get the average of the missing values from the full list and update
        # that value in the missing list
        print("match_idx", match_idx)
        for idx in indx:
            avg = []
            for j in match_idx:
                avg.append(none_lst[j][idx])
            missing_lst[i][idx] = np.average(avg)

    # Rebuild the data so that it's in order of when it was read in
    for indx in none_idx:
        idx = 0
        if len(missing_idx) > 0 and missing_idx[0] < indx:
            idx = missing_idx[0]
            final_data.append(missing_lst[idx])
            missing_lst.pop(0)
            missing_idx.pop(0)
        else:
            final_data.append(none_lst[idx])

    return final_data


# This function takes the given data set and creates "new" features based on the
# data. This includes a feature for if it's a weekday or weekend, and it also
# goes through the last 24 hours and gets the change in the air quality for
# each of the particulates and makes those a feature themselves. Doing it by
# the hour gives the slope (rate of change) over the past hour (rise/run)
def MakeFeatures(dataSet):
    new_data = []
    count = 0
    temp_count = 0
    for data in dataSet:
        curr_data = []
        count = count + 1
        for i in range(0, len(data)):
            if i == 0:
                temp = data[i].split('/')
                curr_data.append(int(temp[1]))
                curr_data.append(int(temp[0]))
                curr_data.append(int(temp[2]))
            else:
                curr_data.append(data[i])

        if count >= 24:
            # Get if it's a weekday [1, 0] or weekend [0, 1]
            dt = datetime.date(curr_data[2], curr_data[0], curr_data[1])
            dt = dt.strftime("%A")
            if dt == "Saturday" or dt == "Sunday":
                curr_data.append(0)
                curr_data.append(1)
            else:
                curr_data.append(1)
                curr_data.append(0)

            # Get the data over the past 24 hours
            past_24 = []
            for i in range(count-24, count-1):
                past_24.append(new_data[i][4:14])

            for i in range(1, len(past_24)):
                past_slope = []
                for j in range(0, len(past_24[i])):
                    temp = past_24[i][j] - past_24[i-1][j]
                    curr_data.append(temp)
        new_data.append(curr_data)
    return new_data[23:]


############################################################################
############################################################################
############################################################################
#####################                       ################################
##################### Begin Main of Program ################################
#####################                       ################################
############################################################################
############################################################################
############################################################################


DIR = os.getcwd()
inFile = DIR + "/RAW_DATA/AirQualityUCI.csv"
outFile = DIR + "AirQuality_clean.csv"
k_nn = 3
num_blank = 2
miss_tkn = "-200"
delim = ';'
dist_eq = "Euclidean"

data = ReadFile(inFile, miss_tkn, num_blank, delim)
print(data[0])

data = MakeFeatures(data)

KNN_Regression(data, int(miss_tkn), k_nn, dist_eq)


#print(data[0:4])

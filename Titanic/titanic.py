from __future__ import print_function, division
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(color_codes=True)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report
# Gets the path/library with my Machine Learning Programs and adds it to the
# current PATH so it can be imported
LIB_PATH = os.path.dirname(os.getcwd()) # Goes one parent directory up
LIB_PATH = LIB_PATH + "/Library/" # Appends the Library folder to the path
sys.path.append(LIB_PATH)

def Plots(dataframe, color_palette):
    ax = sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=dataframe, size=6, kind="bar", legend_out = False, palette=color_palette)
    ax.despine(left=True)
    ax.set_ylabels("Survival Probability")
    ax.set_xlabels("Passenger Class")
    plt.ylim(0, 1.01)
    plt.savefig("class_titanic.png", bbox_inches="tight")
    plt.clf()

    g = sns.PairGrid(data=dataframe.dropna(), hue="Sex", palette=color_palette)
    g.map_diag(plt.hist)
    g.map_offdiag(plt.scatter)
    g.add_legend()
    plt.savefig("scatter_plot.png", bbox_inches="tight")
    plt.clf()

    temp = dataframe.copy(deep=True)
    temp["all"] = ""
    sns.set(font_scale=2.0)
    ax = sns.violinplot(x="all", y="Age", hue="Sex", data=temp, size=4, legend_out = False, split=True, palette=color_palette, inner="quart")
    ax.set_ylabel("Age")
    ax.set_xlabel("Gender")
    plt.savefig("age_titanic.png", bbox_inches="tight")
    plt.clf()

    sns.set(font_scale=1.0)



BASE_DIR = os.getcwd() + "/Raw_Data/"
train_file = BASE_DIR + "train.csv"
test_file = BASE_DIR + "test.csv"

train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

labels = list(train_data.columns.values)
train_ids = list(train_data["PassengerId"])
test_ids = list(test_data["PassengerId"])

lived = train_data.loc[train_data["Survived"] == 1]
died = train_data.loc[train_data["Survived"] == 0]

# Seaborn color palettes
paired = sns.color_palette("Paired")
muted = sns.color_palette("muted")
categorical = sns.color_palette("Set1", n_colors=8, desat=.9)
rgb = sns.color_palette("hls", 8)
ester = sns.color_palette("Set2", 10)
colors = ["windows blue", "red"]
colors = sns.xkcd_palette(colors)

#Plots(train_data, categorical)

mean_age = train_data["Age"].mean()
mean_fare = train_data["Fare"].mean()

train_data["Age"].fillna(mean_age, inplace=True)
train_data["Fare"].fillna(mean_fare, inplace=True)

test_data["Age"].fillna(mean_age, inplace=True)
test_data["Fare"].fillna(mean_fare, inplace=True)

train_set = []
y_vals = []
test_set = []
test_y_vals = []

# Create the feature vectors for the training set
for i in xrange(0, len(train_ids)):
    temp = train_data.iloc[i]
    gen = []
    pclass = [0, 0, 0]
    pclass[int(temp[2]) - 1] = 1
    if temp[4] == "male":
        gen = [1, 0]
    else:
        gen = [0, 1]
    y_vals.append(temp[1])
    train_set.append([pclass[0], pclass[1], pclass[2], gen[0], gen[1], temp[5], temp[6], temp[7], temp[9]])

# Create the feature vectors for the test set
for i in xrange(0, len(test_ids)):
    temp = test_data.iloc[i]
    gen = []
    pclass = [0, 0, 0]
    pclass[int(temp[2]) - 1] = 1
    if temp[4] == "male":
        gen = [1, 0]
    else:
        gen = [0, 1]
    test_y_vals.append(temp[1])
    test_set.append([pclass[0], pclass[1], pclass[2], gen[0], gen[1], temp[5], temp[6], temp[7], temp[9]])

print("Random Forest Classifier")
print("========================")
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(train_set, y_vals)
y_pred = clf.predict(test_set)
print(classification_report(y_pred, test_y_vals))
print('\n')

print("Perceptron")
print("==========")
per = Perceptron()
per = per.fit(train_set, y_vals)
y_pred = per.predict(test_set)
print(classification_report(y_pred, test_y_vals))
print('\n')





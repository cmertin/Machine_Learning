from __future__ import print_function, division
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(color_codes=True)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification

def Plots(dataframe, color_palette):
    # Uses seaborn to plot the class and sex of survivors and deceased
    ax = sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=dataframe, size=6, kind="bar", legend_out = False, palette=color_palette)
    ax.despine(left=True)
    ax.set_ylabels("Survival Probability")
    ax.set_xlabels("Passenger Class")
    plt.ylim(0, 1.01)
    plt.savefig("class_titanic.png", bbox_inches="tight")
    plt.clf()

    # Plots the survival percentage chances based on ages
    max_age = dataframe["Age"].max()
    age_bins = np.arange(0, max_age + 1, 5)
    ages = []
    survived = []
    dead = []
    for i in range(0, len(age_bins)-1):
        age_ = dataframe[dataframe["Age"] >= age_bins[i]]
        age_ = age_[age_["Age"] < age_bins[i + 1]]["Survived"].value_counts()

        try:
            survived_ = age_[1]
        except:
            survived_ = 0
        try:
            dead_ = age_[0]
        except:
            dead_ = 0

        total_ = survived_ + dead_
        ages_temp = "[" + str(age_bins[i]) + ", " + str(age_bins[i+1]) + ")"
        ages.append(ages_temp)

        try:
            survived.append(survived_ / total_)
        except:
            survived.append(0)
        try:
            dead.append(dead_ / total_)
        except:
            dead.append(1)

    plt.plot(survived, label="Survived", color='r')
    plt.plot(dead, label="Dead", color='b')
    plt.xticks(range(len(age_bins)), ages, rotation=70)
    plt.ylabel("Survival Probability")
    plt.xlabel("Age Ranges")
    plt.legend(loc="best")
    plt.savefig("age_titanic.png", bbox_inches="tight")
    plt.clf()

    # Plots the survival chance based on the fares
    max_fare = dataframe["Fare"].max()
    fare_bins = np.arange(0, max_fare + 1, 25)
    survived = []
    dead = []
    fares = []
    for i in range(0, len(fare_bins)-1):
        fare_ = dataframe[dataframe["Fare"] >= fare_bins[i]]
        fare_ = fare_[fare_["Fare"] < fare_bins[i + 1]]["Survived"].value_counts()

        try:
            survived_ = fare_[1]
        except:
            survived_ = 1
        try:
            dead_ = fare_[0]
        except:
            dead_ = 0

        total_ = survived_ + dead_
        fare_temp = "[" + str(fare_bins[i]) + ", " + str(fare_bins[i+1]) + ")"
        fares.append(fare_temp)

        try:
            survived.append(survived_ / total_)
        except:
            survived.append(1)
        try:
            dead.append(dead_ / total_)
        except:
            dead.append(0)

    plt.plot(survived, label="Survived", color='r')
    plt.plot(dead, label="Dead", color='b')
    plt.xticks(range(len(fare_bins)), fares, rotation=70)
    plt.ylabel("Survival Probability")
    plt.xlabel("Fares")
    plt.legend(loc="best")
    plt.savefig("fares_titanic.png", bbox_inches="tight")
    plt.clf()
            
    # Creates a plot based on all of the attributes
    ax = sns.PairGrid(data=dataframe.dropna(), hue="Sex", palette=color_palette)
    ax.map_diag(plt.hist)
    ax.map_offdiag(plt.scatter)
    ax.add_legend()
    plt.savefig("scatter_plot.png", bbox_inches="tight")
    plt.clf()

    # Uses seaborn to create a violin plot on the age distribution on the boat
    temp = dataframe.copy(deep=True)
    temp["all"] = ""
    sns.set(font_scale=2.0)
    ax = sns.violinplot(x="all", y="Age", hue="Sex", data=temp, size=4, legend_out = False, split=True, palette=color_palette, inner="quart", scale="count")
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

# Create the plots
Plots(train_data, categorical)

mean_age = train_data["Age"].mean()
mean_fare = train_data["Fare"].mean()

# Fill empty values with the average from the training set
train_data["Age"].fillna(mean_age, inplace=True)
train_data["Fare"].fillna(mean_fare, inplace=True)

test_data["Age"].fillna(mean_age, inplace=True)
test_data["Fare"].fillna(mean_fare, inplace=True)

train_set = []
y_vals = []
test_set = []
test_y_vals = []

features = ["Class", "Gender", "Age", "SibSp", "ParCh", "Fare"]
# Create the feature vectors for the training set
for i in xrange(0, len(train_ids)):
    temp = train_data.iloc[i]
    gen = []
    pclass = [0, 0, 0]
    pclass[int(temp[2]) - 1] = 1
    if temp[4] == "male":
        gen = 1
    else:
        gen = 0
    y_vals.append(temp[1])
    train_set.append([temp[2], gen, temp[5], temp[6], temp[7], temp[9]])

# Create the feature vectors for the test set
for i in xrange(0, len(test_ids)):
    temp = test_data.iloc[i]
    gen = []
    pclass = [0, 0, 0]
    pclass[int(temp[2]) - 1] = 1
    if temp[4] == "male":
        gen = 1
    else:
        gen = 0
    test_y_vals.append(temp[1])
    test_set.append([temp[2], gen, temp[5], temp[6], temp[7], temp[9]])

# Run the random forest classifier
print("Random Forest Classifier")
print("========================")
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(train_set, y_vals)
y_pred = clf.predict(test_set)
print(classification_report(y_pred, test_y_vals))
print('\n')

# Calculate the important features
importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

print("Feature Ranking:")
X, Y = make_classification(n_samples=1000, n_features=6, n_informative=3, n_redundant=0, n_repeated=0, n_classes=2, random_state=0, shuffle=False)

for f in range(X.shape[1]):
    print("%d. Feature \"%s\":\t (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))


labels = []
for i in xrange(0, len(indices)):
    labels.append(features[indices[i]])

# Plot the important features from greatest to least
plt.figure()
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices], color='r', yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), labels)
plt.xlim([-1, X.shape[1]])
plt.savefig("feature_importance.png", bbox_inches="tight")
plt.savefig("feature_importance.pdf", bbox_inches="tight")
#plt.show()



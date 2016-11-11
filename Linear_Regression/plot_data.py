from __future__ import print_function, division
import matplotlib.pyplot as plt

def ReadData(filename):
    data = []

    lines = [line.rstrip('\n') for line in open(filename)]

    for line in lines:
        data.append(float(line))

    return data

x_file = "age.dat"
y_file = "height.dat"
outfile = "linreg_1.png"

x = ReadData(x_file)
y = ReadData(y_file)

plt.scatter(x, y)
plt.xlabel("Ages in Years")
plt.ylabel("Height in Meters")
plt.savefig(outfile, type="png", bbox_inches="tight")

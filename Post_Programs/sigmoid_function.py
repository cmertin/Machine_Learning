from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np

def SigmoidFunction(x):
    denominator = 1 + np.exp(-x)
    return (1/denominator)

x = np.arange(-6, 6, 0.001)
y = []

for x_ in x:
    y.append(SigmoidFunction(x_))

plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Sigmoid Function")
plt.grid()
plt.yticks(np.arange(0, 1, 0.1))
plt.savefig("Sigmoid_Function.png", bbox_inches="tight")
plt.show()

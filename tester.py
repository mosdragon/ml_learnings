import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from perceptron import Perceptron

X = None
y = None

def init():
    df = pd.read_csv("./iris.data")
    # Select setosa and versicolor
    global y
    global X
    y = df.iloc[0:100, 4].values
    y = np.where(y == "Iris-setosa", -1, 1)
    X = df.iloc[0:100, [0, 1, 2, 3]].values

def error_plot(errors):
    plt.plot(range(1, len(errors) + 1), errors, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of updates')
    plt.show()

def use_perceptron():
    pc = Perceptron()
    errors = pc.fit(X, y)
    error_plot(errors)


if __name__ == "__main__":
    init()
    use_perceptron()

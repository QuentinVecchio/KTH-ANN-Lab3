import hopfield
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

def scenario3_1():
    x1=[-1, -1, 1, -1, 1, -1, -1, 1]
    x2=[-1, -1, -1, -1, -1, 1, -1, -1]
    x3=[-1, 1, 1, -1, -1, 1, -1, 1]

    X = np.stack((x1, x2, x3))

    x1d=[1, -1, 1, -1, 1, -1, -1, 1]
    x2d=[1, 1, -1, -1, -1, 1, -1, -1]
    x3d=[1, 1, 1, -1, 1, 1, -1, 1]

    Xd = np.stack((x1d, x2d, x3d))

    HF = hopfield.HopfieldNetwork(pattern_number=8)

    print("Store points")
    HF.store(X)
    print("Original point")
    print(X - HF.littleModel(X))
    print("Distored points")
    print(X - HF.littleModel(Xd))

    print("Find fixed point")
    Xf = HF.searchFixedPoint(Xd)
    print(X - Xf)

    HF.searchAllFixedPoint()

    x1=[1, 1, -1, -1, -1, -1, 1, 1]
    x2=[-1, -1, 1, 1, 1, -1, 1, -1]

    print(HF.searchFixedPoint(np.stack((x1, x2))))


def scenario3_2():
    with open("../pict.dat", "r") as f:
        line = f.readlines()[0]
    X = np.array(list(map(int, line.split(","))))
    X = X.reshape((11,1024))

    for i in range(11):
        printFigure(X[i], str(i+1))

    HF = hopfield.HopfieldNetwork(pattern_number=1024)
    HF.store(X[:3])

    X3 = HF.littleModel(X[:3])

    print("Check X3 fixed points")
    print(np.sum(X3 - X[:3], axis=1))

    a = HF.littleModel(X[9:10])
    b = HF.littleModel(X[10:11])

    #printFigure(a, "oo10")
    #printFigure(b, "oo11")

    x = np.random.choice([-1, 1], 1024)
    printFigure(x, "randomInit")
    x1 = HF.searchFixedPoint(x)
    printFigure(x1, "random")
    print("JE COMPRENDS PAS LA DERNIRE QUESTION ICI")

def printFigure(X, title):
    A = X.reshape((32, 32))
    plt.matshow(A)
    plt.savefig(title + ".png")

def scenario3_3():
    with open("../pict.dat", "r") as f:
        line = f.readlines()[0]
    X = np.array(list(map(int, line.split(","))))
    X = X.reshape((11,1024))

    HF = hopfield.HopfieldNetwork(pattern_number=1024)
    HF.store(X[:3])

    print(HF.E(X[0]))
    print(HF.E(X[1]))
    print(HF.E(X[2]))

    print(HF.E(np.random.choice([-1, 1], 1024)))
    print(HF.E(np.random.choice([-1, 1], 1024)))
    print(HF.E(np.random.choice([-1, 1], 1024)))


def scenario3_4():
    print("TODO")

def scenario3_5():
    print("TODO")

def scenario3_6():
    print("TODO")

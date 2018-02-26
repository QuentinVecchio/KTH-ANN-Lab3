import hopfield
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import random as rd
import copy


def scenario3_1():
    x1 = [-1, -1, 1, -1, 1, -1, -1, 1]
    x2 = [-1, -1, -1, -1, -1, 1, -1, -1]
    x3 = [-1, 1, 1, -1, -1, 1, -1, 1]

    X = np.stack((x1, x2, x3))

    x1d = [1, -1, 1, -1, 1, -1, -1, 1]
    x2d = [1, 1, -1, -1, -1, 1, -1, -1]
    x3d = [1, 1, 1, -1, 1, 1, -1, 1]

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

    x1 = [1, 1, -1, -1, -1, -1, 1, 1]
    x2 = [-1, -1, 1, 1, 1, -1, 1, -1]

    print(HF.searchFixedPoint(np.stack((x1, x2))))


def scenario3_2():
    with open("../pict.dat", "r") as f:
        line = f.readlines()[0]
    X = np.array(list(map(int, line.split(","))))
    X = X.reshape((11, 1024))

    for i in range(11):
        saveFigure(X[i], str(i + 1))

    HF = hopfield.HopfieldNetwork(pattern_number=1024)
    HF.store(X[:3])

    X3 = HF.littleModel(X[:3])

    print("Check X3 fixed points")
    print(np.sum(X3 - X[:3], axis=1))

    a = HF.sequentialConvergence(X[9])
    b = HF.sequentialConvergence(X[10])

    saveFigure(a, "oo10")
    saveFigure(b, "oo11")

    x = np.random.choice([-1, 1], 1024)
    saveFigure(x, "randomInit")
    xs = HF.sequentialUpdate100(x)
    for i in range(len(xs)):
        saveFigure(xs[i], "random" + str(i))


def saveFigure(X, title):
    A = X.reshape((32, 32))
    plt.matshow(A)
    plt.savefig(title + ".png")
    plt.close()


def scenario3_3():
    with open("../pict.dat", "r") as f:
        line = f.readlines()[0]
    X = np.array(list(map(int, line.split(","))))
    X = X.reshape((11, 1024))

    HF = hopfield.HopfieldNetwork(pattern_number=1024)
    HF.store(X[:3])

    print("Energy attractors")

    print(HF.E(X[0]))
    print(HF.E(X[1]))
    print(HF.E(X[2]))

    print("Energy random points")

    print(HF.E(np.random.choice([-1, 1], 1024)))
    print(HF.E(np.random.choice([-1, 1], 1024)))
    print(HF.E(np.random.choice([-1, 1], 1024)))

    print("Energy over iterations")

    HF.sequentialUpdatePrintE(np.random.choice([-1, 1], 1024))

    print("Energy over iterations for random W")
    W = np.random.normal(0, 1, 1024 * 1024).reshape((1024, 1024))

    HF.setW(W)
    HF.sequentialUpdatePrintE(np.random.choice([-1, 1], 1024))

    print("Energy over iterations for random W but symetric")

    HF.setW(0.5 * (W + W.T))
    HF.sequentialUpdatePrintE(np.random.choice([-1, 1], 1024))

    print("WHATS HAPPEN ????")


def scenario3_4():
    with open("../pict.dat", "r") as f:
        line = f.readlines()[0]
    X = np.array(list(map(int, line.split(","))))
    X = X.reshape((11, 1024))

    HF = hopfield.HopfieldNetwork(pattern_number=1024)
    HF.store(X[:3])

    p1 = X[0]
    p2 = X[1]
    p3 = X[2]

    noise = [0, 0.1, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1]
    l = np.arange(1024)
    np.random.shuffle(l)

    for x in noise:
        p1d = copy.deepcopy(X[0])
        p2d = copy.deepcopy(X[1])
        p3d = copy.deepcopy(X[2])
        print("")
        print("noise: " + str(x))

        for i in range(int(x * 1024)):
            p1d[l[i]] *= -1
            p2d[l[i]] *= -1
            p3d[l[i]] *= -1

        a = HF.littleModel(p1d)
        b = HF.littleModel(p2d)
        c = HF.littleModel(p3d)

        #print(a - p1)
        print("-  Difference p1")
        m = sum(a - p1)
        print(m)
        if m != 0:
            a1 = HF.searchFixedPoint(p1d)
            if (np.array_equal(a1, p1)):
                print("converge to right attractor")

        print("-  Difference p2")
        m = sum(b - p2)
        print(m)
        if m != 0:
            b1 = HF.searchFixedPoint(p2d)
            if (np.array_equal(b1, p2)):
                print("converge to right attractor")

        print("-  Difference p3")
        m = sum(c - p3)
        print(m)
        if m != 0:
            c1 = HF.searchFixedPoint(p3d)
            if (np.array_equal(c1, p3)):
                print("converge to right attractor")


def scenario3_5_1():
    with open("../pict.dat", "r") as f:
        line = f.readlines()[0]
    X = np.array(list(map(int, line.split(","))))
    X = X.reshape((11, 1024))

    HF = hopfield.HopfieldNetwork(pattern_number=1024)

    # we will check the performance on a 20% noised picture
    l = np.arange(1024)
    np.random.shuffle(l)

    for i in range(10):
        print("Number of stored images " + str(i))
        HF.store(X[:i])

        noised = copy.deepcopy(X[:i])

        for i in range(len(noised)):
            for j in range(int(0.2 * 1024)):
                noised[i][l[j]] *= -1

        for i in range(len(noised)):
            temp = HF.littleModel(noised[i])
            if (not(np.array_equal(temp, X[i]))):
                print("Recover error.")

    X = (np.random.randint(2, size=1024*50)*2-np.ones(1024*50)).reshape((50, 1024))
    for i in range(50):
        print("Number of stored images " + str(i))
        HF.store(X[:i])

        noised = copy.deepcopy(X[:i])

        for i in range(len(noised)):
            for j in range(int(0.2 * 1024)):
                noised[i][l[j]] *= -1

        for i in range(len(noised)):
            temp = HF.littleModel(noised[i])
            if (not(np.array_equal(temp, X[i]))):
                print("Recover error.")

def scenario3_5_2():
    nbError = []
    HF = hopfield.HopfieldNetwork(pattern_number=100)
    X = (np.random.randint(2, size=300*100)*2-np.ones(300*100)).reshape((300, 100))
    for i in range(100):
        print("Number of stored images " + str(i))
        HF.store(X[:i])

        noised = copy.deepcopy(X[:i])
        nb = 0
        for i in range(len(noised)):
            temp = HF.littleModel(noised[i])
            if (np.array_equal(temp, X[i])):
                nb+=1
        nbError.append(nb)

    plt.plot(nbError)
    plt.title("Number of stable patterns vs the number of learned pattern")
    plt.show()
    print(nbError)

    nbError = []
    l = np.arange(100)
    np.random.shuffle(l)
    for i in range(100):
        print("Number of stored images " + str(i))
        HF.store(X[:i])

        noised = copy.deepcopy(X[:i])
        for i in range(len(noised)):
            for j in range(int(0.1 * 100)):
                noised[i][l[j]] *= -1
        nb = 0
        for i in range(len(noised)):
            temp = HF.littleModel(noised[i])
            if (np.array_equal(temp, X[i])):
                nb+=1
        nbError.append(nb)

    plt.plot(nbError)
    plt.title("Number of stable patterns vs the number of learned pattern with noise")
    plt.show()
    print(nbError)

    nbError = []
    HF = hopfield.HopfieldNetwork(pattern_number=100)
    X = (np.random.randint(2, size=300*100)*2-np.ones(300*100)).reshape((300, 100))
    for i in range(100):
        print("Number of stored images " + str(i))
        HF.store(X[:i])
        HF.removeDiag()

        noised = copy.deepcopy(X[:i])
        nb = 0
        for i in range(len(noised)):
            temp = HF.littleModel(noised[i])
            if (np.array_equal(temp, X[i])):
                nb+=1
        nbError.append(nb)

    plt.plot(nbError)
    plt.title("Number of stable patterns vs the number of learned pattern no diag")
    plt.show()
    print(nbError)

    nbError = []
    l = np.arange(100)
    np.random.shuffle(l)
    for i in range(100):
        print("Number of stored images " + str(i))
        HF.store(X[:i])
        HF.removeDiag()

        noised = copy.deepcopy(X[:i])
        for i in range(len(noised)):
            for j in range(int(0.1 * 100)):
                noised[i][l[j]] *= -1
        nb = 0
        for i in range(len(noised)):
            temp = HF.littleModel(noised[i])
            if (np.array_equal(temp, X[i])):
                nb+=1
        nbError.append(nb)

    plt.plot(nbError)
    plt.title("Number of stable patterns vs the number of learned pattern with noise no diag with bias")
    plt.show()
    print(nbError)


    nbError = []
    X = np.sign(np.random.rand(300*100)-0.25*np.ones(300*100)).reshape((300, 100))
    print(X)
    for i in range(100):
        print("Number of stored images " + str(i))
        HF.store(X[:i])

        noised = copy.deepcopy(X[:i])
        nb = 0
        for i in range(len(noised)):
            temp = HF.littleModel(noised[i])
            if (np.array_equal(temp, X[i])):
                nb+=1
        nbError.append(nb)

    plt.plot(nbError)
    plt.title("Number of stable patterns vs the number of learned pattern with bias")
    plt.show()
    print(nbError)

    nbError = []
    l = np.arange(100)
    np.random.shuffle(l)
    for i in range(100):
        print("Number of stored images " + str(i))
        HF.store(X[:i])

        noised = copy.deepcopy(X[:i])
        for i in range(len(noised)):
            for j in range(int(0.1 * 100)):
                noised[i][l[j]] *= -1
        nb = 0
        for i in range(len(noised)):
            temp = HF.littleModel(noised[i])
            if (np.array_equal(temp, X[i])):
                nb+=1
        nbError.append(nb)

    plt.plot(nbError)
    plt.title("Number of stable patterns vs the number of learned pattern with noise")
    plt.show()
    print(nbError)

def scenario3_6():
    print("TODO")

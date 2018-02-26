import numpy as np
import itertools

class HopfieldNetwork:
    def __init__(self, pattern_number):
        self.pattern_number = pattern_number
        self.W = np.zeros((pattern_number, pattern_number))
        print("Init Hopfield")

    def littleModel(self, X):
        X = np.dot(X, self.W)
        try:
            for i in range(len(X)):
                for j in range(len(X[i])):
                    n = X[i][j]
                    if n >= 0:
                        X[i][j] = 1
                    else:
                        X[i][j] = -1
        except:
            for i in range(len(X)):
                n = X[i]
                if n >= 0:
                    X[i] = 1
                else:
                    X[i] = -1
        return X

    def computeW(self, X):
        W = np.dot(X.T, X)
        print("W computed.")
        return W

    def store(self, X):
        np.set_printoptions(threshold=np.inf)

        self.W = self.computeW(X)
        #print(self.W)

    def searchFixedPoint(self, X):
        X = self.littleModel(X)
        fixed = False
        i = 1
        while not(fixed) and i < 100:
            i += 1
            X_T = self.littleModel(X)
            fixed = np.array_equal(X_T, X)
            X = X_T
        print(str(i-1) + " steps.")
        return X

    def searchAllFixedPoint(self):
        permut = ["".join(seq) for seq in itertools.product("01", repeat=self.pattern_number)]
        X = np.zeros((len(permut), self.pattern_number))
        for i in range(len(permut)):
            for j in range(self.pattern_number):
                X[i][j] = int(permut[i][j].replace("0", "-1"))


        points = self.searchFixedPoint(X)
        points = np.unique(points, axis=0)
        print(str(len(points)) + " fixed points.")
        print(points)

    def setW(self, W):
        self.W = W

    def E(self, x):
        e = 0
        for i in range(self.pattern_number):
            for j in range(self.pattern_number):
                e += self.W[i][j] * x[i] * x[j]
        return -e

import copy
import numpy as np
from sklearn import datasets


def distance(method):
    if method == 1:
        # Euclidean distances
        for i in range(c):
            for k in range(n):
                dis = 0
                for j in range(nd):
                    dis += (data[k][j] - v[i][j]) ** 2
                d[i][k] = dis ** 0.5
    elif method == 2:
        # Manhattan distances
        for i in range(c):
            for k in range(n):
                dis = 0
                for j in range(nd):
                    dis += abs(data[k][j]-v[i][j])
                d[i][k] = dis
    elif method == 3:
        # Chebyshev distances
        for i in range(c):
            for k in range(n):
                dis = []
                for j in range(nd):
                    dis.append(abs(data[k][j] - v[i][j]))
                d[i][k] = max(dis)
    elif method == 4:
        # Minkowski distances
        for i in range(c):
            for k in range(n):
                dis = 0
                for j in range(nd):
                    dis += abs(data[k][j] - v[i][j]) ** p
                d[i][k] = dis ** (1/p)


if __name__ == '__main__':

    # Load data
    iris = datasets.load_iris()
    data = iris.data
    n, nd = data.shape

    c = int(input("Enter Number of clusters: "))
    mPrime = float(input("Enter value of m': "))
    e = float(input("Enter value of tolerance: "))

    v = np.zeros((c, nd), dtype=float)
    d = np.zeros((c, n), dtype=float)
    u = np.zeros((c, n), dtype=float)
    uPrime = np.full((c, n), -1, dtype=float)

    # initialize u
    j = i = 0
    while j < n:
        u[i][j] = 1
        j += 1
        i += 1
        if i == c:
            i = 0

    # select distance calculation method
    select = int(input("Enter distance calculation method:\n"
                       "Euclidean 1\n"
                       "Manhattan 2\n"
                       "Chebyshev 3\n"
                       "Minkowski 4\n"))
    if select == 4:
        p = float(input("Enter Minkowski parameter: "))

    loop = 0
    while True:
        loop += 1
        # calculate centers
        for i in range(c):
            for k in range(nd):
                s = m = 0
                for j in range(n):
                    s += (u[i][j] ** mPrime) * data[j][k]
                    m += u[i][j] ** mPrime
                v[i][k] = s / m if m > 0 else 0

        # calculate distances
        distance(select)

        # calculate membership values
        for i in range(c):
            for k in range(n):
                mu = 0
                for j in range(c):
                    if uPrime[i][k] == -1:
                        if d[i][k] == 0:
                            uPrime[i][k] = 1
                            break
                        if d[j][k] == 0:
                            uPrime[i][k] = 0
                            break
                    mu += (d[i][k] / d[j][k]) ** (2 / (mPrime - 1))
                else:
                    uPrime[i][k] = 1 / mu

        # check tolerance
        tol = []
        for i in range(c):
            for j in range(n):
                tol.append(abs(u[i][j] - uPrime[i][j]))
        if max(tol) <= e:
            print("tolerance = {:.4f} in loop {}".format(max(tol), loop))
            break
        else:
            u = copy.deepcopy(uPrime)
            print("tolerance = {:.4f} in loop {}".format(max(tol), loop))

    np.set_printoptions(precision=2, suppress=True)
    print(uPrime)

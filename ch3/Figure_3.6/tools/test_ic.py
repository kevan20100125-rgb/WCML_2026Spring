import numpy as np
import numpy.linalg as la

def icholesky(a):
    n = a.shape[1]
    eps = 1 / 32
    for j in range(n):
        a[j, j] = np.sqrt(a[j, j])
        for i in range(j+1, n):
            if a[i, j] != 0:
                a[i, j] = a[i, j] / a[j, j]

        for k in range(j+1, n):
            for i in range(k+1, n):
                if a[i, k] != 0:
                    a[i, k] = a[i, k] - a[i, j] * a[k, j]

    for i in range(n):
        for j in range(i+1, n):
            a[i, j] = 0

    return a

a = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
aha = np.conj(a).T @ a

l = icholesky(aha)
print(l)
# -*- coding:utf-8 -*-
# @Time  : 2021/9/10 20:12
# @Author: STARain
# @File  : swamp.py

from pylab import *
from .utils import NLE


def amp(x, A, y, sigma2, t_max, mu=2, orth=False):
    m, n = A.shape
    sqrA = A * A

    # initialization
    ave, var = np.zeros((n, 1)), np.ones((n, 1))
    w, v = y, np.ones((m, 1))

    mse = np.zeros(t_max, dtype=np.float64)
    for t in range(t_max):
        w = A.dot(ave) - sqrA.dot(var) * (y - w) / (sigma2 + v)
        v = sqrA.dot(var)
        s = 1. / sqrA.T.dot(1. / (sigma2 + v))
        r = ave + s * A.T.dot((y - w) / (sigma2 + v))

        ave, var, _, _ = NLE(s, r, orth=orth, mu=mu, EP=True)

        # Uncomment this line to learn \Delta iteratively.
        # d = d * sum(((y - w) / (d + v)) ** 2) / sum(1. / (d + v))

        mse[t] = np.mean((x-ave)**2) / np.mean(x**2)

    return ave, mse


def amp_mimo(x, A, y, sigma2, t_max, mu=2, orth=False):

    m, n = A.shape
    # initialization
    z = np.zeros((n, 1))
    r = y
    tau = n / m / sigma2 / 2
    xhat = np.zeros((n, 1))
    mse = np.zeros(t_max, dtype=np.float64)

    for t in range(t_max):
        z = xhat + A.T @ r
        xhat, tau, _, _  = NLE(sigma2*(1+tau), z, orth=orth, mu=mu)
        tau_new = n / m / sigma2 * tau
        r = y - A @ xhat + tau_new / (1+tau) * r
        tau = tau_new
        mse[t] = np.mean((x - xhat) ** 2) / np.mean(x ** 2)

    return xhat, mse


def swamp(x, A, y, sigma2, t_max, mu=2, orth=False, sparse=False, pnz=.1):
    m, n = A.shape
    sqrA = A * A

    # initialization
    ave, var = np.zeros((n, 1)), np.ones((n, 1))
    r, s = np.zeros((n, 1)), np.zeros((n, 1))
    w, v = y, np.ones((m, 1))

    mse = np.zeros(t_max, dtype=np.float64)
    for t in range(t_max):
        g = (y - w) / (sigma2 + v)  # (m, 1)
        v = sqrA.dot(var)  # (m, 1)
        w = A.dot(ave) - v * g  # (m, 1)
        # sequential update
        for i in permutation(n):
            s[i] = 1. / sqrA[:, i].dot(1 / (sigma2 + v))
            r[i] = ave[i] + s[i] * A[:, i].dot((y - w) / (sigma2 + v))
            # store the previous value
            ave_old, var_old = ave[i].copy(), var[i].copy()
            v_old = v.copy()
            # NLE  todo: var use expectation or estimation?; orth or not?
            ave[i], var[i], _, _ = NLE(s[i], r[i], orth=orth, mu=mu, sparse=sparse, pnz=pnz)
            # update w and v, the difference is vert important for the robustness of the algorithm!!!
            v += sqrA[:, i].reshape(-1, 1)  * (var[i] - var_old)  # (m, 1), same diff for mu in 1-m
            w += A[:, i].reshape(-1, 1) * (ave[i] - ave_old) - g * (v - v_old)  # (m, 1)

        # sigma2 = sigma2 * sum(((y - w) / (sigma2 + v)) ** 2) / sum(1. / (sigma2 + v))

        # todo: learning sigma2 iteratively?
        mse[t] = np.mean((x-ave)**2) / np.mean(x**2)

    return ave, mse


def prior(r, s, t):
    rho, m_pr, v_pr = t

    m_eff = (m_pr * s + r * v_pr) / (s + v_pr)
    v_eff = v_pr * s / (s + v_pr)
    z = ((1. - rho) / rho) * sqrt(v_pr / v_eff) * \
        exp(-.5 * (r ** 2 / s - (m_pr - r) ** 2 / (s + v_pr)))

    # x_post and v_post, no_orth
    a = m_eff / (z + 1)
    c = z * a ** 2 + v_eff / (z + 1)

    return a, c


def swamp_o(x, A, y, sigma2, t_max, mu=2, orth=False, sparse=False, pnz=.1):
    m, n = A.shape
    sqrF = A * A
    x = x.reshape(-1)
    y = y.reshape(-1)
    a, c = zeros(n), .5 * ones(n)
    r, s = zeros(n), zeros(n)
    w, v = y, ones(m)

    mse = np.zeros(t_max, dtype=np.float64)
    for t in range(t_max):
        g = (y - w) / (sigma2 + v)
        v = sqrF.dot(c)
        w = A.dot(a) - v * g
        for i in permutation(n):
            s[i] = 1. / sqrF[:, i].dot(1. / (sigma2 + v))
            r[i] = a[i] + s[i] * A[:, i].dot((y - w) / (sigma2 + v))
            a_old, c_old = a[i].copy(), c[i].copy()
            # a[i], c[i] = prior(r[i], s[i], array([pnz, 0.0, 1.0]))
            a[i], c[i], _, _ = NLE(s[i], r[i], orth=orth, mu=mu, sparse=sparse, pnz=pnz)
            v_old = v.copy()
            v += sqrF[:, i].dot(c[i] - c_old)
            w += A[:, i].dot(a[i] - a_old) - (v - v_old) * g

        # Uncomment this line to learn \Delta iteratively.
        sigma2 = sigma2 * sum(((y - w) / (sigma2 + v)) ** 2) / sum(1. / (sigma2 + v))

        mse[t] = np.mean((x-a)**2) / np.mean(x**2)

    return a.reshape(-1,1), mse

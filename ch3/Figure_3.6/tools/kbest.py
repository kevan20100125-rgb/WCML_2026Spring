# -*- coding:utf-8 -*-
# @Time  : 2023/5/8 17:26
# @Author: STARain
# @File  : kbest.py


import numpy as np
import numpy.linalg as la


def kbest(x, H, y, mu=2, k=32):
    # initialization
    n = H.shape[1]  # 2 * nt
    cons_size = 2 ** (mu // 2)
    cons = np.arange(-(cons_size - 1), cons_size - 1 + 1, 2) / np.sqrt(2 * (cons_size ** 2 - 1) / 3)
    q, r = la.qr(H)  # (2 * nr, 2 * nr) and (2 * nr, 2 * nt)
    y_pre = (q[:, :n]).T @ y  # (2nt, 1)
    r = r[:n, :]  # (2nt, 2nt)

    cur_siv = np.zeros((n, k))  # current survivor path
    cur_ped = np.zeros(k)  # current partial euclidean distance
    next_cons = np.tile(cons, (k, ))  # (cons_size * k), the constellation point after extending
    next_ped = np.zeros((cons_size * k, ))  # the PED after extending

    temp = (y_pre[-1] - r[-1, -1] * cons) ** 2  # (cons_size, )
    index = np.argsort(temp)
    temp = temp[index]  #
    n_cand = min(cons_size, k)
    cur_siv[n - 1, :n_cand] = cons[index[:n_cand]]  # the k best candidates
    cur_ped[:n_cand] = temp[:n_cand]  # current PED

    for i in range(n - 2, -1, -1):  # n-2 ~ 0
        for j in range(n_cand):
            next_ped[j * cons_size:(j + 1) * cons_size] = (y_pre[i] - r[i, :] @ cur_siv[:, j:(j + 1)]
                                                           - r[i, i] * cons) ** 2 + cur_ped[j]

        # sort -- for the k best candidates
        n_hypo = n_cand * cons_size  # number of hypothesis
        tmp = next_ped[:n_hypo]
        index = np.argsort(tmp)
        hypo_ped = tmp[index]
        n_cand = min(n_hypo, k)

        # update
        pos = (np.floor(index[:n_cand] / cons_size)).astype(int)  # determine the index of the k best candidate points (mod cons_size)
        cur_siv[:, :n_cand] = cur_siv[:, pos]  # (2nt, n_cand), pos: 0~n_cand-1, extend from which path
        cur_siv[i, :n_cand] = next_cons[index[:n_cand]]  # survivor for the n-th dimension/sublattice
        cur_ped[:n_cand] = hypo_ped[:n_cand]

    # select the sample that minimize the ML cost
    xhat = cur_siv[:, 0:1]  # (2nt, 1)
    mse = np.mean(abs(x - xhat) ** 2)
    return xhat, mse
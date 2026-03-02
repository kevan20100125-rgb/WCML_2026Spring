#!/usr/bin/python

import numpy as np
import numpy.linalg as la
from .utils import de2bi, _QPSK_Constellation, _16QAM_Constellation, _64QAM_Constellation


def SISO_STS_SD(A, y, mu, snr, app_llr, preprocess, clipping='legacy', max_llr=0.4):
    """
    SISO Single Tree Search (STS) Sphere Decoder (SD)
    Args:
        A (): channel matrix
        y (): receive signal vector
        mu (): modulation order
        snr (): SNR
        app_llr (): a-posterior LLR
        preprocess (): type of QR decomposition preprocessing for SD detector
        clipping (): type of LLR clipping, 'exact' or 'legacy', default 'legacy'
        max_llr (): maximum LLR for clipping, default 0.4

    Returns:

    """
    # preprocessing
    m = A.shape[0]  # Rx
    n = A.shape[1]  # Tx
    if preprocess == 'QRD':  # standard QR decomposition
        q, r = la.qr(A)  # shape(m,m) and shape(m,n)
        order = np.arange(n)  # (4,)
    elif preprocess == 'SQRD':  # perform SQRD according to [Wuebben et al, 2003]
        q, r, order = sqr(A.copy())
    elif preprocess == 'MMSEQRD':  # regularized QRD
        A_aug = np.vstack((A, np.sqrt(n / snr) * np.eye(n)))
        q_aug, r_aug = la.qr(A_aug)
        q = q_aug[:m, :n].copy()  # extract upper left sub-matrix
        r = r_aug[:n, :].copy()  # extract upper half
        order = np.arange(n)
    elif preprocess == 'MMSESQRD':  # regularized SQRD
        A_aug = np.vstack((A, np.sqrt(n / snr) * np.eye(n)))
        q_aug, r_aug, order = sqr(A_aug)
        q = q_aug[:m, :n].copy()  # extract upper left sub-matrix
        r = r_aug[:n, :].copy()  # extract upper half
    else:
        raise RuntimeError('No valid detector')

    y_hat = np.conj(q).T @ y  # shape(m,1)
    # invert LLR definition
    app_llr = - app_llr  # shape(n, mu)

    # a priori information
    app_llr_ordered = app_llr[order, :]  # shape(n, mu)
    app_bias = np.sum(abs(app_llr_ordered), axis=1).T  # compute optimum max-log LLR bias, shape(1, n)

    # initialization
    radius = np.inf
    path_hist = np.zeros((n, 1), dtype=int)  # path history
    st = np.zeros((n, 2 ** mu))  # stack

    # soft output specific
    bin_array = np.fliplr(de2bi(np.arange(2 ** mu), mu))  # reshape from (2^mu, ) to (2^mu, mu)
    x_ml = np.inf * np.ones((n, mu))  # ML hypothesis
    lambda_ml = np.inf  # metric of ML hypothesis
    lambda_mlbar = np.inf * np.ones((n, mu))  # metric of counter hypotheses

    if mu == 2:
        constellation_norm = _QPSK_Constellation / np.sqrt(2)
    elif mu == 4:
        constellation_norm = _16QAM_Constellation / np.sqrt(10)
    elif mu == 6:
        constellation_norm = _64QAM_Constellation / np.sqrt(42)
    else:
        raise RuntimeError('Modulation order not supported')

    # add root node to stack
    level = n - 1
    app_inc = 0.5 * (app_bias[level])  # shape(2^mu, 1)
    st[level, :] = (abs(y_hat[level] - r[level, level] * constellation_norm) ** 2).reshape(-1) + app_inc  # shape (1,16)
    expanded_nodes = 0  # not count the root node

    # begin sphere decoder
    while level <= n - 1:
        # find smallest PED in boundary
        min_ped, idx = min(st[level, :]), np.argmin(st[level, :])
        # only proceed if list is not empty
        if min_ped < np.inf:
            st[level, idx] = np.inf  # mark child as tested
            new_path = np.concatenate((np.array([[idx]]),
                                       path_hist[level+1:].reshape(-1, 1)))  # new best path
            # calculate current pruning radius
            radius = 0
            for lev in range(len(new_path) - 1, -1, -1):  # rule for known bits
                bits = bin_array[(new_path[lev]).T, :].copy().reshape(-1)
                for bit in range(mu):
                    if bits[bit] == x_ml[lev+n-len(new_path), bit]:  # ML-hypothesis
                        radius = max(radius, lambda_ml)
                    else:  # counter hypothesis (bits ~= xML)
                        if bits[bit] == 0:  # +1
                            radius = max(radius,
                                         lambda_mlbar[lev+n-len(new_path), bit]-
                                         app_llr_ordered[lev+n-len(new_path), bit])
                        if bits[bit] == 1:  # -1
                            radius = max(radius,
                                         lambda_mlbar[lev+n-len(new_path), bit]+
                                         app_llr_ordered[lev+n-len(new_path), bit])
            for lev in range(n-len(new_path)):  # rule for unknown bits
                for bit in range(mu):
                    if x_ml[lev, bit] == 0:  # +1
                        radius = max(radius, lambda_mlbar[lev, bit]+app_llr_ordered[lev, bit])
                    if x_ml[lev, bit] == 1:  # -1
                        radius = max(radius, lambda_mlbar[lev, bit]-app_llr_ordered[lev, bit])
            # search child
            if min_ped < radius:
                # valid candidate found
                expanded_nodes = expanded_nodes + 1
                if level > 0:
                    # expand this best node
                    path_hist[level:, 0] = new_path.copy().reshape(-1)
                    level -= 1  # downstep
                    df = r[level, level+1:] @ constellation_norm[path_hist[level+1:, 0]]
                    app_inc = 0.5 * (app_bias[level] * np.ones((len(constellation_norm), 1)) -
                                     np.sign(0.5 - bin_array) @ np.conj(app_llr_ordered[level, :]).reshape(-1, 1))
                    st[level, :] = (min_ped + abs(y_hat[level] - r[level, level] * constellation_norm - df) ** 2
                                    + app_inc).reshape(-1)
                else:
                    # valid leaf found
                    x = bin_array[(new_path).reshape(-1), :].copy()
                    # update hypothesis and clip LLRs
                    if min_ped < lambda_ml:  # case 1)
                        for lev in range(n):
                            for bit in range(mu):
                                if x[lev, bit] != x_ml[lev, bit]:
                                    if x_ml[lev, bit] == 0:  # +1
                                        lambda_mlbar[lev, bit] = lambda_ml + app_llr_ordered[lev, bit]
                                    if x_ml[lev, bit] == 1:  # -1
                                        lambda_mlbar[lev, bit] = lambda_ml - app_llr_ordered[lev, bit]
                        lambda_ml = min_ped  # update ML metric
                        x_ml = x.copy()  # update ML hypothesis
                        # clipping (corrected)
                        if clipping == 'exact':  # yields prefectly clipped LLRs
                            for lev in range(n):
                                for bit in range(mu):
                                    lambda_mlbar[lev, bit] = max(lambda_ml - max_llr,
                                                                 min(lambda_mlbar[lev, bit], lambda_ml+max_llr))
                        elif clipping == 'legacy':  # low-complexity approximation
                            for lev in range(n):
                                for bit in range(mu):
                                    lambda_mlbar[lev, bit] = min(lambda_mlbar[lev, bit], lambda_ml+max_llr)
                        else:
                            raise RuntimeError('No valid clipping')
                    else:  # case 2)
                        for lev in range(n):
                            for bit in range(mu):
                                if x[lev, bit] != x_ml[lev, bit]:
                                    if (x_ml[lev, bit] == 0 and
                                            min_ped - app_llr_ordered[lev, bit] < lambda_mlbar[lev, bit]):
                                        lambda_mlbar[lev, bit] = min_ped - app_llr_ordered[lev, bit]
                                    if (x_ml[lev, bit] == 1 and
                                            min_ped + app_llr_ordered[lev, bit] < lambda_mlbar[lev, bit]):  # -1
                                        lambda_mlbar[lev, bit] = min_ped + app_llr_ordered[lev, bit]
        else:  # no more childs to be checked
            level += 1

    # compute soft outputs and reorder LLRs
    bin_array_coded = np.sign(0.5 - x_ml) * (lambda_mlbar - lambda_ml)  # (n, mu)
    # if early termination happens, additional clipping might be required
    bin_array_coded = np.maximum(np.minimum(bin_array_coded, max_llr), -max_llr)
    llr_e1_data = np.zeros((n, mu))
    llr_e1_data[order.T, :] = bin_array_coded.copy()
    # invert LLR definition
    llr_e1_data = - llr_e1_data

    return llr_e1_data, expanded_nodes


# sorted QR decomposition [Wuebben et al., 2003]
def sqr(A):
    m = A.shape[0]  # Rx
    n = A.shape[1]  # Tx
    q, r = A, np.zeros((n, n), dtype=complex)
    p = np.eye(n, dtype=int)
    for i in range(n):
        s = np.diag(np.conj(q[:, i:].T) @ q[:, i:])
        pp, idx = np.amin(s), np.argmin(s)
        idx = idx + i
        tmp = q[:, i].copy()
        q[:, i], q[:, idx] = q[:, idx], tmp
        tmp = p[:, i].copy()
        p[:, i], p[:, idx] = p[:, idx], tmp
        tmp = r[:, i].copy()
        r[:, i], r[:, idx] = r[:, idx], tmp
        r[i, i] = np.sqrt(s[idx - i])
        q[:, i] = q[:, i] / r[i, i]
        for k in range(i + 1, n):
            r[i, k] = np.conj(q[:, i].T) @ q[:, k]
            q[:, k] = q[:, k] - r[i, k] * q[:, i]
    qa = q.copy()
    r = r[:n, :]  # extract upper half only (lower half is zero)
    order = np.arange(n) @ p

    return qa, r, order

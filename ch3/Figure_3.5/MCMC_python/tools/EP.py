#!/usr/bin/python

import sys
import numpy as np
import numpy.linalg as la
import math
from .utils import QAM_Modulation, NLE, de2bi, _QPSK_Constellation, _16QAM_Constellation, _64QAM_Constellation


def EP(x,A,y,noise_var,T=10,mu=2,soft=False,pp_llr=None):  # ub as output, stable
    # T = T+1
    # initialize
    M = A.shape[0]
    N = A.shape[1]
    # gamma = np.zeros((N,1))
    # Lambda = 1 / (np.ones(N)/2)
    beta = 0.2
    AT = A.T
    ATA = AT@A
    MSE = np.zeros(T)
    if pp_llr is None:
        pp_llr = np.zeros((mu//2, N))
    else:
        pp_llr = np.concatenate((pp_llr[:,:mu//2], pp_llr[:,mu//2:]), axis=0)
    bin_array = np.sign(de2bi(np.arange(2 ** (mu // 2)), mu//2) - 0.5).astype(np.int)  # (2 ** mu, mu)
    if mu == 2:  # (0 1) --> (-1 +1)
        constellation_norm = np.array([-1, +1]) / np.sqrt(2)
    elif mu == 4:
        constellation_norm = np.array([-3, -1, +3, +1]) / np.sqrt(10)
    else:
        constellation_norm = np.array([-7, -5, -1, -3, +7, +5, +1, +3]) / np.sqrt(42)

    # calculate soft estimates-- mean and  variance of constellation
    dist = 0.5 * bin_array @ pp_llr  # (2**(mu//2), N)
    dist += np.amin(dist, axis=0)
    probs = np.exp(dist).T  # (N, 2**(mu//2))
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    s_est = np.sum(probs * constellation_norm, axis=1, keepdims=True)  # (N, 1)
    e_est = np.sum(probs * (s_est * np.ones((N, 2 ** (mu // 2))) - constellation_norm) ** 2, axis=1,
                   keepdims=True)  # (N, 1)
    e_est = np.maximum(e_est, 1e-8)

    # calculate the initial pair for EP
    Lambda = (1 / e_est).reshape(N)  # (N,)
    gamma = s_est * Lambda.reshape(N,1)  # (N, 1)

    for t in range(T):
        # compute the mean and covariance matrix
        Sigma = la.inv(ATA + noise_var * np.diag(Lambda))
        Mu = Sigma @ (AT@y + noise_var * gamma)
        MSE[t] = np.mean((x-Mu)**2)

        # compute the extrinsic mean and covariance matrix
        diag = noise_var * np.diag(Sigma).reshape(N,1)
        vab = diag/(1 - diag * Lambda.reshape(N,1))
        vab = np.maximum(vab, 5e-7)
        uab = vab*(Mu/diag - gamma)

        # compute the posterior mean and covariance matrix
        if soft:
            _, _, ub, vb, ext_probs = NLE(vab, uab, orth=False, mu=mu, EP=True, norm=np.sqrt(1), soft=True)
            ext_probs = np.maximum(np.exp(-(uab*np.ones((N, 2 ** (mu//2))) - constellation_norm) ** 2 / (2*vab)), 1e-100)
            post_probs = probs * ext_probs
            post_probs = post_probs / np.sum(post_probs, axis=1, keepdims=True)  # (N, 2 ** (mu//2))
            ub = np.sum(post_probs * constellation_norm, axis=1, keepdims=True)
            vb = np.sum(post_probs * (ub * np.ones((N, 2 ** (mu//2))) - constellation_norm) ** 2, axis=1, keepdims=True)
        else:
            _, _, ub, vb = NLE(vab,uab,orth=False,mu=mu,EP=True,norm=np.sqrt(1))
        vb = np.maximum(vb,5e-13)

        # update gamma and Lambda
        gamma_last = gamma
        Lambda_last = Lambda
        gamma = (ub*vab - uab*vb) / vb / vab
        Lambda = ((vab -vb) / vb / vab).reshape(N)
        idx = Lambda < 0
        Lambda[idx] = Lambda_last[idx]
        gamma[idx] = gamma_last[idx]
        # damping
        gamma = beta*gamma +(1-beta) *gamma_last
        Lambda = beta*Lambda +(1-beta) *Lambda_last
        # if t >= 1:
        #     if MSE[t] > MSE[t-1]:
        #         pass
    if soft:
        return ub, MSE, ext_probs
    return ub, MSE

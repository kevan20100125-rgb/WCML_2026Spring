#!/usr/bin/python

import sys
import numpy as np
import numpy.linalg as la
import math
from .utils import QAM_Modulation, NLE, de2bi, _QPSK_Constellation, _16QAM_Constellation, _64QAM_Constellation

beta = np.zeros(20)
para = {}
# loading the trained model parameters
try:
    for k,v in np.load("EP_4×4_16QAM_15dB_I_1.npz").items():
        para[k] = v
except IOError:
    print("no such file")
    pass
# get parameters for CG-OAMP-NET
for t in range(20):
    if para.get("beta_"+str(t)+":0",-1) != -1:
        beta[t] = para["beta_"+str(t)+":0"]
beta = 1. / (1. + np.exp(-beta))


def EP(x,A,y,noise_var,T=10,mu=2,soft=False,pp_llr=None):  # ub as output, stable

    # ─── YOUR CODE HERE ──────────────────────────────────────────────────── #
    #
    #
    #
    # ─────────────────────────────────────────────────────────────────────── #

    return ub, MSE

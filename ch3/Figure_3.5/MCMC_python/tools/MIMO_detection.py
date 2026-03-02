#!/usr/bin/python
from __future__ import division
import numpy as np
import os
import time
import sys
import math
import numpy.linalg as la
from scipy.linalg import toeplitz, sqrtm, cholesky, dft
from .utils import QAM_Modulation, QAM_Demodulation, indicator, lmmse_ce
from .EP import EP
from .MHGD import mhgd, nag_mcmc
from commpy import QAMModem

from gurobipy import *
import scipy.io as sio

pi = math.pi


def MIMO_detection_simulate(sysin, SNR=40):
    Mr, Nt, mu = sysin.Mr, sysin.Nt, sysin.mu
    channel_type, rho_tx, rho_rx = sysin.channel_type, sysin.rho_tx, sysin.rho_rx
    csi = sysin.csi
    sysin.acc, sysin.avg_ng, sysin.avg_ns, sysin.avg_nz = 0, 0, 0, 0
    err_bits_target = 1000
    total_err_bits = 0
    total_bits = 0
    ser = 0
    count = 0
    start = time.time()
    MSE = 0
    total_time = 0.
    if mu > 6:
        modem = QAMModem(2 ** mu)
    else:
        modem = None
    if channel_type == 'corr':
        sqrtRtx, sqrtRrx = corr_channel(Mr, Nt, rho_tx=rho_tx, rho_rx=rho_rx)

    if csi == 2:
        Np = Nt
        wlmmse, xp = channel_est(sysin, SNR)
    complex_dict = {'MHGD', 'NAG_MCMC'}

    num_trail = 10000
    # while True:
    for i in range(num_trail):
        np.random.seed(count)  # choose the same channel
        bits = np.random.binomial(n=1, p=0.5, size=(Nt * mu,))  # label
        bits_mod = QAM_Modulation(bits, mu, modem=modem)
        x = bits_mod.reshape(Nt, 1)

        H = np.sqrt(1 / 2 / Mr) * (np.random.randn(Mr, Nt)
                                   + 1j * np.random.randn(Mr, Nt))  # Rayleigh MIMO channel
        if channel_type == 'corr':  # Correlated MIMO channel
            H = sqrtRrx @ H @ sqrtRtx

        # channel input & output
        y = H @ x

        signal_power = Nt / Mr  # signal power per receive ant.: E(|xi|^2)=1 E(||hi||_F^2)=Nt
        sigma2 = signal_power * 10 ** (-(SNR) / 10)  # noise power per receive ant.; average SNR per receive ant.
        noise = np.sqrt(sigma2 / 2) * (np.random.randn(Mr, 1) + 1j * np.random.randn(Mr, 1))
        y = y + noise

        if csi == 2:
            noise = np.sqrt(sigma2 / 2) * (np.random.randn(Mr, Np) + 1j * np.random.randn(Mr, Np))
            yp = H @ xp + noise
            yp_vec = yp.reshape(-1, 1)
            h_hat = wlmmse @ yp_vec
            H = h_hat.reshape(Mr, Nt)

        if sysin.detect_type not in complex_dict:
            # convert complex into real
            x = np.concatenate((np.real(x), np.imag(x)))
            H = np.concatenate((np.concatenate((np.real(H), -np.imag(H)), axis=1),
                                np.concatenate((np.imag(H), np.real(H)), axis=1)))
            y = np.concatenate((np.real(y), np.imag(y)))

        sta = time.time()
        x_hat, MSE = detector(sysin, H, x, y, sigma2, MSE, modem=modem)
        end = time.time()

        if sysin.detect_type not in complex_dict:
            # back into np.complex64
            x_hat = x_hat.reshape((2, Nt))
            x_hat = x_hat[0, :] + 1j * x_hat[1, :]

        # Demodulate
        x_hat_demod = QAM_Demodulation(x_hat, mu, modem)

        if i >= 1:
            total_time += (end - sta)

        # calculate BER
        err_bits = np.sum(np.not_equal(x_hat_demod, bits))
        total_err_bits += err_bits
        total_bits += mu * Nt
        count = count + 1
        if err_bits > 0:
            ser += calc_ser(x_hat_demod, bits, Nt, mu)
            sys.stdout.write(
                '\rtotal_err_bits={teb} total_bits={tb} BER={BER:.9f} SER={SER:.6f} ng={ng:.3f} ns={ns:.3f} '
                'nz={nz:.3f} acc={acc:.3f}'
                .format(teb=total_err_bits, tb=total_bits, BER=total_err_bits / total_bits,
                        SER=ser / count / Nt, ng=sysin.avg_ng / count, ns=sysin.avg_ns / count, nz=sysin.avg_nz / count,
                        acc=sysin.acc / count))
            sys.stdout.flush()
        # if total_err_bits > err_bits_target or total_bits > 1e7:
        if i == num_trail - 1:
            end = time.time()
            iter_time = end - start
            print("\nSNR=", SNR, "iter_time:", iter_time)
            print("average detection time:", total_time / (num_trail - 1))
            ber = total_err_bits / total_bits
            ser = ser / count / Nt
            print("BER:", ber)
            print("SER:", ser)
            print("MSE:", 10 * np.log10(MSE / count))
            print("Average ng:", sysin.avg_ng / count)
            print("Average ns:", sysin.avg_ns / count)
            print("Average nz:", sysin.avg_nz / count)
            print("Average acc rate:", sysin.acc / count)
            break

    return ber, ser, np.array([total_err_bits, total_bits, sysin.avg_ns / count, sysin.avg_nz / count])


def calc_ser(x_hat_demod, bits, Nt, mu):
    # ser = 0
    # for n in range(Nt):
    #     err_bits = np.sum(np.not_equal(x_hat_demod[n*mu:(n+1)*mu], bits[n*mu:(n+1)*mu]))
    #     if err_bits > 0:
    #         ser += 1
    err = np.not_equal(x_hat_demod, bits).reshape(Nt, mu)
    ser = np.sum(np.any(err, axis=1))
    return ser


def corr_channel(Mr, Nt, rho_tx=0.5, rho_rx=0.5):
    Rtx_vec = np.ones(Nt)
    for i in range(1, Nt):
        Rtx_vec[i] = rho_tx ** i
    Rtx = toeplitz(Rtx_vec)
    if Mr == Nt and rho_tx == rho_rx:
        Rrx = Rtx
    else:
        Rrx_vec = np.ones(Mr)
        for i in range(1, Mr):
            Rrx_vec[i] = rho_rx ** i
        Rrx = toeplitz(Rrx_vec)

    # another way of constructing kronecker model
    # C = cholesky(np.kron(Rtx,Rrx))    # complex correlation
    # C = sqrtm(np.sqrt(np.kron(Rtx, Rrx)))  # power field correlation--what's an equivalent model?
    # return C

    sqrtRtx = sqrtm(Rtx)  # sqrt decomposition for power field

    if Mr == Nt and rho_tx == rho_rx:
        sqrtRrx = sqrtRtx
    else:
        sqrtRrx = sqrtm(Rrx)

    return sqrtRtx, sqrtRrx


def detector(sys, H, x, y, sigma2, MSE, modem=None):
    detect_type = sys.detect_type
    Mr, Nt = sys.Mr, sys.Nt
    mu = sys.mu
    T = sys.T
    if detect_type == 'ZF':  # ZF
        HT = H.T
        x_hat = la.inv(HT @ H) @ HT @ y
        MSE += np.mean(abs(x - x_hat) ** 2)
    elif detect_type == 'MMSE':  # MMSE
        HT = H.T
        x_hat = la.inv(HT @ H + sigma2 / 2 * np.eye(2 * Nt)) @ HT @ y
        # MSE += np.mean(abs(x - x_hat) ** 2)
        MSE += 0.
    elif detect_type == 'EP':
        x_hat, mse = EP(x, H, y, sigma2 / 2, T=T, mu=mu)
        MSE += mse
    elif detect_type == 'ML':
        x_hat = mlSolver(y, H, mu).reshape(-1, 1)
        MSE += np.mean((x - x_hat) ** 2)
    elif detect_type == 'MHGD':  # todo
        x_hat, mse = mhgd()
        MSE += mse
    elif detect_type == 'Gibbs':  # todo
        x_hat, mse = gibbs()
        MSE += mse
    elif detect_type == 'NAG_MCMC':
        x_hat, mse, ns = nag_mcmc(x, H, y, sigma2, mu=mu, iter=sys.samples, samplers=sys.samplers, ng=sys.ng,
                                  mmse_init=sys.mmse_init,
                                  quantize=sys.quantize, post=sys.post, sur=sys.sur, early_stop=sys.es,
                                  constellation_norm=modem.constellation if mu > 6 else None)
        MSE += mse
        sys.avg_ns += ns
    else:
        raise RuntimeError('The selected detector does not exist!')

    return x_hat, MSE


def mlSolver(y, h_real, mu):
    # status = []
    m, n = h_real.shape[0], h_real.shape[1]
    model = Model('mimo')
    M = 2 ** (mu // 2)
    sigConst = np.linspace(-M + 1, M - 1, M)
    sigConst /= np.sqrt((sigConst ** 2).mean())
    sigConst /= np.sqrt(2.)  # Each complex transmitted signal will have two parts
    z = model.addVars(n, M, vtype=GRB.BINARY, name='z')
    s = model.addVars(n, ub=max(sigConst) + .1, lb=min(sigConst) - .1, name='s')
    e = model.addVars(m, ub=200.0, lb=-200.0, vtype=GRB.CONTINUOUS, name='e')
    model.update()

    ### Constraints and variables definitions
    # define s[i]
    for i in range(n):
        model.addConstr(s[i] == quicksum(z[i, j] * sigConst[j] for j in range(M)))
    # constraint on z[i,j]
    model.addConstrs((z.sum(j, '*') == 1 for j in range(n)), name='const1')
    # define e
    for i in range(m):
        e[i] = quicksum(h_real[i, j] * s[j] for j in range(n)) - y[i]

    ### define the objective function
    obj = e.prod(e)
    model.setObjective(obj, GRB.MINIMIZE)
    model.Params.logToConsole = 0
    model.setParam('TimeLimit', 100)
    model.update()

    model.optimize()

    # retrieve optimization result
    solution = model.getAttr('X', s)
    # status.append(model.getAttr(GRB.Attr.Status) == GRB.OPTIMAL)
    # print(GRB.OPTIMAL, model.getAttr(GRB.Attr.Status))
    if model.getAttr(GRB.Attr.Status) == 9:
        print(np.linalg.cond(h_real))
    x_hat = []
    for num in solution:
        x_hat.append(solution[num])
    return np.array(x_hat)


def channel_est(sysin, snr):
    Mr, Nt, mu = sysin.Mr, sysin.Nt, sysin.mu
    channel_type, rho_tx, rho_rx = sysin.channel_type, sysin.rho_tx, sysin.rho_rx
    Np = Nt  # the number of pilot vectors
    if channel_type == 'corr':
        sqrtRtx, sqrtRrx = corr_channel(Mr, Nt, rho_tx=rho_tx, rho_rx=rho_rx)
    print('calculate covariance and LMMSE weight matrix for CE')
    num = 10000
    rhh = np.zeros((Mr * Nt, Mr * Nt), dtype=complex)
    for n in range(num):
        H = np.sqrt(1 / 2 / Mr) * (np.random.randn(Mr, Nt)
                                   + 1j * np.random.randn(Mr, Nt))  # Rayleigh MIMO channel
        if channel_type == 'corr':  # Correlated MIMO channel
            H = sqrtRrx @ H @ sqrtRtx
        h = H.reshape(-1, 1)
        rhh += h @ np.conj(h.T)
    rhh /= num
    xp = dft(Np)[:Nt, :]  # orthogonal pilots (nt, np)
    sigma2 = 10 ** (-snr / 10)
    yp = np.zeros((Mr, Np), dtype=complex)
    wlmmse = lmmse_ce(xp, yp, sigma2, rhh)
    print('end')
    return wlmmse, xp


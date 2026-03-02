
import numpy as np
import numpy.linalg as la
from .utils import _QPSK_Constellation, _16QAM_Constellation, _64QAM_Constellation
from scipy.linalg import cholesky, sqrtm
from scipy.special import softmax
import time


# samples from multiple categorical distributions simultaneously: https://github.com/numpy/numpy/issues/15201
def categorical(p):
    return (p.cumsum(-1) >= np.random.uniform(size=p.shape[:-1])[..., None]).argmax(-1)


def dis_ula(x, A, y, noise_var, mu=2, iter=8, samplers=16, lr_approx=False, mmse_init=False, vec_step_size=True,
            constellation_norm=None, hessian_approx=False):
    # initialization
    mr, nt = A.shape
    dqam = np.sqrt(3 / 2 / (2 ** mu - 1))  # Eavg = 2/3*(M-1)
    AH = (A).T
    AHA = AH @ A
    covar = np.eye(nt)
    if mu == 2:
        constellation_norm = np.array([-1, +1]) * dqam
    elif mu == 4:
        constellation_norm = np.array([-3, -1, +1, +3]) * dqam
    elif mu == 6:
        constellation_norm = np.array([-7, -5, -3, -1, +1, +3, +5, +7]) * dqam
    else:
        constellation_norm = constellation_norm * dqam

    if mmse_init is True:
        x_mmse = la.inv(AHA + noise_var * np.eye(nt)) @ AH @ y
        xhat = constellation_norm[np.argmin(abs(x_mmse * np.ones((nt, 2 ** (mu // 2))) - constellation_norm),
                                              axis=1)].reshape(-1, 1)  # quantization
        xhat = np.tile(xhat, (samplers, 1, 1))  # (np, nt, 1)
    else:
        xhat = constellation_norm[np.random.randint(low=0, high=2 ** (mu // 2), size=(samplers, nt, 1))].copy()
    r = y - A @ xhat  # (np, nr, 1)
    r_norm = (np.transpose(r, axes=[0, 2, 1]) @ r)  # (np, 1, 1)
    x_survivor, r_norm_survivor = xhat.copy(), r_norm.copy()

    L = la.norm(AHA, 'fro')
    lr = 1 / L
    lr = 0.5

    # def cat_dist_sample(prob):
    #     sample = np.random.choice(a=constellation_norm, p=prob, size=(1, ))
    #     return sample

    for t in range(iter):
        grad = (AH @ r)  # / (2 * noise_var)  # (np, nt, 1)
        diff = constellation_norm - xhat * np.ones((1, 1, 2 ** (mu // 2)))
        first_term = grad * diff  # (np, nt, 2 ** mu)
        second_term = diff ** 2 / (2 * lr)
        cat_dist = softmax(first_term - second_term, axis=2)  # (np, nt, 2 ** mu)
        # x_prop = np.apply_along_axis(cat_dist_sample, axis=2, arr=cat_dist)  # (np, nt, 1) np.random.choice is slow
        x_prop = constellation_norm[categorical(cat_dist)[:, :, np.newaxis]]  # (np, nt, 1)
        r_prop = y - A @ x_prop
        r_norm_prop = np.transpose(r_prop, axes=[0, 2, 1]) @ r_prop  # (np, 1, 1)
        update = np.squeeze(r_norm_survivor > r_norm_prop)
        if update.any():
            x_survivor[update] = x_prop[update]
            r_norm_survivor[update] = r_norm_prop[update]
        xhat, r = x_prop.copy(), r_prop.copy()
    # select the sample that minimizes the ML cost
    mse = np.squeeze(np.mean((x - x_survivor) ** 2, axis=1))
    if 0. not in mse:
        a = None
    x_hat = x_survivor[np.argmin(r_norm_survivor), :, :].copy()

    return x_hat, mse


def dis_mala(x, A, y, noise_var, mu=2, iter=8, samplers=16, lr_approx=False, mmse_init=False, vec_step_size=False,
             constellation_norm=None, hessian_approx=False):
    # initialization
    mr, nt = A.shape
    dqam = np.sqrt(3 / 2 / (2 ** mu - 1))  # Eavg = 2/3*(M-1)
    AH = (A).T
    AHA = AH @ A
    grad_preconditioner = la.inv(AHA + noise_var / (dqam ** 2) * np.eye(nt))
    covar = np.eye(nt)
    # alpha =  1 / ((nt / 8) ** (1 / 3))
    alpha = 0.79
    if mu == 2:
        constellation_norm = np.array([-1, +1]) * dqam
    elif mu == 4:
        constellation_norm = np.array([-3, -1, +1, +3]) * dqam
    elif mu == 6:
        constellation_norm = np.array([-7, -5, -3, -1, +1, +3, +5, +7]) * dqam
    else:
        constellation_norm = constellation_norm * dqam

    if mmse_init is True:
        x_mmse = la.inv(AHA + noise_var * np.eye(nt)) @ AH @ y
        xhat_idx = np.argmin(abs(x_mmse * np.ones((1, 2 ** (mu // 2))) - constellation_norm),
                                              axis=1)  # (nt, )
        xhat_idx = np.tile(xhat_idx[:, np.newaxis], (samplers, 1, 1))
        xhat = constellation_norm[xhat_idx]  # quantization
    else:
        xhat_idx = np.random.randint(low=0, high=2 ** (mu // 2), size=(samplers, nt, 1))
        xhat = constellation_norm[xhat_idx].copy()
    r = y - A @ xhat  # (np, nr, 1)
    r_norm = (np.transpose(r, axes=[0, 2, 1]) @ r)  # (np, 1, 1)
    x_survivor, r_norm_survivor = xhat.copy(), r_norm.copy()

    # lr = 0.5  # 0.9 for unity eigenvalue preconditioner

    # def cat_dist_sample(prob):
    #     idx = np.random.choice(a=2 ** (mu // 2), p=prob, size=(1, ))
    #     return idx

    step_size = np.maximum(dqam, np.sqrt(r_norm / mr)) * alpha  # (np, 1, 1)
    # step_size = 0.05
    scale = 1
    acc_rate = 0
    grad = grad_preconditioner @ (AH @ r)  # / (2 * noise_var)  # (np, nt, 1)
    diff = constellation_norm - xhat * np.ones((1, 1, 2 ** (mu // 2)))
    first_term = grad * diff / step_size ** 2  # (np, nt, 2 ** mu)
    second_term = diff ** 2 / (2 * step_size ** 2)
    cat_dist = softmax((first_term - second_term) / scale, axis=2)  # (np, nt, 2 ** mu)
    # delta = first_term - second_term
    # cat_dist = delta - np.amax(np.abs(delta), axis=2, keepdims=True)
    for t in range(iter):
        # x_prop_idx = np.apply_along_axis(cat_dist_sample, axis=2, arr=cat_dist)  # (np, nt, 1) np.random.choice is slow
        x_prop_idx = categorical(cat_dist)[:, :, np.newaxis]  # (np, nt, 1)
        x_prop = constellation_norm[x_prop_idx]
        r_prop = y - A @ x_prop
        r_norm_prop = np.transpose(r_prop, axes=[0, 2, 1]) @ r_prop  # (np, 1, 1)
        update = np.squeeze(r_norm_survivor > r_norm_prop)
        if update.any():
            x_survivor[update] = x_prop[update]
            r_norm_survivor[update] = r_norm_prop[update]

        # step_size = np.maximum(dqam, np.sqrt(r_norm_prop / mr)) * alpha  # (np_update, 1, 1)
        # acceptance step
        p_forward_fac = np.take_along_axis(cat_dist, x_prop_idx, axis=2)  # (np, nt, 1)
        # lp_forward = np.sum(np.log(p_forward_fac), axis=1)[:, :, np.newaxis]  # (np, 1, 1)
        # lp_forward = np.sum(p_forward_fac, axis=1)[:, :, np.newaxis]
        lp_forward = np.log(np.prod(p_forward_fac, axis=1))[:, :, np.newaxis]  # (np, 1, 1)
        grad_rev = grad_preconditioner @ (AH @ r_prop)  # / (2 * noise_var)
        diff_rev = constellation_norm - x_prop * np.ones((1, 1, 2 ** (mu // 2)))
        first_term_rev = grad_rev * diff_rev / step_size ** 2  # (np, nt, 2 ** mu)
        second_term_rev = diff_rev ** 2 / (2 * step_size ** 2)
        cat_dist_rev = softmax((first_term_rev - second_term_rev) / scale, axis=2)  # (np, nt, 2 ** mu)
        p_reverse_fac = np.take_along_axis(cat_dist_rev, xhat_idx, axis=2)  # (np, nt, 1)
        # lp_reverse = np.sum(np.log(p_reverse_fac), axis=1)[:, :, np.newaxis]  # (np, 1, 1) collapse in high-dimensional
        # delta = first_term_rev - second_term_rev
        # cat_dist_rev = delta - np.amax(np.abs(delta), axis=2, keepdims=True)
        # p_reverse_fac = np.take_along_axis(cat_dist_rev, xhat_idx, axis=2)  # (np, nt, 1)
        # lp_reverse = np.sum(p_reverse_fac, axis=1)[:, :, np.newaxis]
        lp_reverse = np.log(np.prod(p_reverse_fac, axis=1))[:, :, np.newaxis]  # (np, 1, 1)
        log_pacc = np.minimum(0, - (r_norm_prop - r_norm) / (2 * noise_var) + lp_reverse - lp_forward)
        p_acc = np.exp(log_pacc)
        p_uni = np.random.uniform(low=0.0, high=1.0, size=(samplers, 1, 1))
        index = np.squeeze(p_acc >= p_uni, axis=(1, 2))
        acc_rate += sum(index)
        if index.any():
            xhat[index], r[index], r_norm[index], xhat_idx[index], cat_dist[index] = \
                x_prop[index], r_prop[index], r_norm_prop[index], x_prop_idx[index], cat_dist_rev[index]
            # xhat[index], r[index], r_norm[index], xhat_idx[index] = \
            #     x_prop[index], r_prop[index], r_norm_prop[index], x_prop_idx[index]
            step_size[index] = np.maximum(dqam, np.sqrt(r_norm[index] / mr)) * alpha  # (np_update, 1, 1)

    # select the sample that minimizes the ML cost
    mse = np.squeeze(np.mean((x - x_survivor) ** 2, axis=1))
    if 0. not in mse:
        a = None
    x_hat = x_survivor[np.argmin(r_norm_survivor), :, :].copy()

    return x_hat, mse, (acc_rate / samplers / iter)


def dis_pavg(x, A, y, noise_var, mu=2, iter=8, samplers=16, lr_approx=False, mmse_init=False, vec_step_size=False,
             constellation_norm=None, hessian_approx=False):
    # initialization
    mr, nt = A.shape
    dqam = np.sqrt(3 / 2 / (2 ** mu - 1))  # Eavg = 2/3*(M-1)
    AH = (A).T
    AHA = AH @ A
    # alpha = 1 / ((nt / 8) ** (1 / 3))
    alpha = 1
    grad_preconditioner = - (AHA / 1)
    lam = - np.amin(la.eigvalsh(grad_preconditioner))
    step_size = 2
    deps = (lam + 2 / step_size)
    sigma_eps_root = sqrtm(grad_preconditioner + deps * np.eye(nt))
    # sigma_eps_root = np.eye(nt)
    if mu == 2:
        constellation_norm = np.array([-1, +1]) * dqam
    elif mu == 4:
        constellation_norm = np.array([-3, -1, +1, +3]) * dqam
    elif mu == 6:
        constellation_norm = np.array([-7, -5, -3, -1, +1, +3, +5, +7]) * dqam
    else:
        constellation_norm = constellation_norm * dqam

    if mmse_init is True:
        x_mmse = la.inv(AHA + noise_var * np.eye(nt)) @ AH @ y
        xhat = constellation_norm[np.argmin(abs(x_mmse * np.ones((nt, 2 ** (mu // 2))) - constellation_norm),
                                              axis=1)].reshape(-1, 1)  # quantization
        xhat = np.tile(xhat, (samplers, 1, 1))  # (np, nt, 1)
    else:
        xhat_idx = np.random.randint(low=0, high=2 ** (mu // 2), size=(samplers, nt, 1))
        xhat = constellation_norm[xhat_idx].copy()
    r = y - A @ xhat  # (np, nr, 1)
    r_norm = (np.transpose(r, axes=[0, 2, 1]) @ r)  # (np, 1, 1)
    x_survivor, r_norm_survivor = xhat.copy(), r_norm.copy()

    # def cat_dist_sample(prob):
    #     idx = np.random.choice(a=2 ** (mu // 2), p=prob, size=(1, ))
    #     return idx

    # step_size = np.maximum(dqam, np.sqrt(r_norm / mr)) * alpha  # (np, 1, 1)
    scale = 1
    acc_rate = 0
    for t in range(iter):
        current_mean = sigma_eps_root @ xhat
        z = np.random.randn(samplers, nt, 1) + current_mean  # should be computed each step  todo:can it be reused?
        grad = (AH @ r) / (1)  # (np, nt, 1)
        # todo: how to give a large proportion to the gradient
        first_term = (grad - grad_preconditioner @ xhat + sigma_eps_root @ z) * np.ones((1, 1, 2 ** (mu // 2))) \
                     * constellation_norm  # (np, nt, 2 ** mu)
        second_term = constellation_norm ** 2 * deps / 2
        cat_dist = softmax((first_term - second_term) / scale, axis=2)  # (np, nt, 2 ** mu)
        # x_prop_idx = np.apply_along_axis(cat_dist_sample, axis=2, arr=cat_dist)  # (np, nt, 1) np.random.choice is slow
        x_prop_idx = categorical(cat_dist)[:, :, np.newaxis]  # (np, nt, 1)
        x_prop = constellation_norm[x_prop_idx]
        r_prop = y - A @ x_prop
        r_norm_prop = np.transpose(r_prop, axes=[0, 2, 1]) @ r_prop  # (np, 1, 1)
        update = np.squeeze(r_norm_survivor > r_norm_prop)
        if update.any():
            x_survivor[update] = x_prop[update]
            r_norm_survivor[update] = r_norm_prop[update]

        # acceptance step
        p_forward_fac = np.take_along_axis(cat_dist, x_prop_idx, axis=2)  # (np, nt, 1)
        lp_forward = np.sum(np.log(p_forward_fac), axis=1)[:, :, np.newaxis]  # (np, 1, 1)
        grad_rev = (AH @ r_prop) / (1)
        first_term_rev = (grad_rev - grad_preconditioner @ x_prop + sigma_eps_root @ z) * np.ones((1, 1, 2 ** (mu // 2))) \
                       * constellation_norm  # (np, nt, 2 ** mu)
        second_term_rev = constellation_norm ** 2 * deps / 2
        cat_dist_rev = softmax((first_term_rev - second_term_rev) / scale, axis=2)  # (np, nt, 2 ** mu)
        p_reverse_fac = np.take_along_axis(cat_dist_rev, xhat_idx, axis=2)  # (np, nt, 1)
        lp_reverse = np.sum(np.log(p_reverse_fac), axis=1)[:, :, np.newaxis]  # (np, 1, 1)
        current_gauss_res, proposal_gauss_res = z - current_mean, z - sigma_eps_root @ x_prop  # z - chol_factor @ st
        current_gauss_term = np.transpose(current_gauss_res, axes=[0, 2, 1]) @ current_gauss_res / 2  # (np, 1, 1)
        proposal_gauss_term = np.transpose(proposal_gauss_res, axes=[0, 2, 1]) @ proposal_gauss_res / 2
        log_pacc = np.minimum(0, - (r_norm_prop - r_norm) / (4 * noise_var) + (lp_reverse - lp_forward) +
                              current_gauss_term - proposal_gauss_term)
        p_acc = np.exp(log_pacc)
        p_uni = np.random.uniform(low=0.0, high=1.0, size=(samplers, 1, 1))
        index = np.squeeze(p_acc >= p_uni, axis=(1, 2))
        acc_rate += sum(index)
        if index.any():
            xhat[index], r[index], r_norm[index], xhat_idx[index], cat_dist[index] = \
                x_prop[index], r_prop[index], r_norm_prop[index], x_prop_idx[index], cat_dist_rev[index]
            # step_size[index] = np.maximum(dqam, np.sqrt(r_norm[index] / mr)) * alpha  # (np_update, 1, 1)

    # select the sample that minimizes the ML cost
    mse = np.squeeze(np.mean((x - x_survivor) ** 2, axis=1))
    if 0. not in mse:
        a = None
    x_hat = x_survivor[np.argmin(r_norm_survivor), :, :].copy()

    return x_hat, mse, (acc_rate / samplers / iter)
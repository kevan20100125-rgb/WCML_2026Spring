
import numpy as np
import numpy.linalg as la
import math


def hmc(x, A, y, noise_var, mu=2, iter=8, samplers=16, leapfrogs=10, mmse_init=False):
    # initialization
    mr, nt = A.shape
    dqam = np.sqrt(3 / 2 / (2 ** mu - 1))  # Eavg = 2/3*(M-1)
    AT = A.T
    if mu == 2:
        constellation_norm = np.array([-1, +1]) / np.sqrt(2)
        sigma = 0.3
    elif mu == 4:
        constellation_norm = np.array([-3, -1, +3, +1]) / np.sqrt(10)
        sigma = 0.15
    else:
        constellation_norm = np.array([-7, -5, -1, -3, +7, +5, +1, +3]) / np.sqrt(42)
        sigma = 0.075

    if mmse_init is True:
        x_mmse = la.inv(AT @ A + noise_var * np.eye(nt)) @ AT @ y
        xhat = constellation_norm[np.argmin(abs(x_mmse * np.ones((nt, 2 ** (mu // 2))) - constellation_norm),
                                              axis=1)].reshape(-1, 1)  # quantization
        # xhat = constellation_norm[np.random.randint(low=0, high=2 ** mu, size=(samplers, nt, 1))].copy()
        # xhat[np.random.randint(low=0, high=samplers)] = xmmse  # only one sampler initialized by MMSE
        xhat = np.tile(xhat, (samplers, 1, 1))  # (np, nt, 1)
    else:
        xhat = constellation_norm[np.random.randint(low=0, high=2 ** (mu // 2), size=(samplers, nt, 1))].copy()
    r = y - A @ xhat  # (np, nr, 1)
    r_norm = np.transpose(r, axes=[0, 2, 1]) @ r  # (np, 1, 1)
    x_survivor, r_norm_survivor = xhat.copy(), r_norm.copy()

    step_size = 0.3 / nt
    ones = np.ones((samplers, nt, 2 ** (mu // 2)))
    for t in range(iter):
        p = np.random.randn(samplers, nt, 1)  # zero-mean, unit-variance
        current_p = p.copy()
        q = xhat.copy()

        # make a half step for momentum at the beginning
        u_prior, gradu_prior = log_density(q, constellation_norm, mu, sigma)
        r = y - A @ q
        u_likelihood = np.transpose(r, axes=[0, 2, 1]) @ r / (2 * noise_var)
        gradu_likelihood = - AT @ r / noise_var  # todo: remove noise_var
        gradu = gradu_prior + gradu_likelihood
        current_u = u_prior + u_likelihood
        p = p - step_size * gradu / 2

        # Alternate full steps for position and momentum
        for j in range(leapfrogs):
            # make a full step for the position
            q = q + step_size * p

            # update survivor
            xtmp = constellation_norm[np.argmin(abs(q * ones - constellation_norm),
                                                  axis=2)].reshape(-1, nt, 1)  # quantization
            rtmp = y - A @ xtmp
            r_norm_tmp = np.transpose(rtmp, axes=[0, 2, 1]) @ rtmp
            update = np.squeeze(r_norm_survivor > r_norm_tmp)
            if update.any():
                x_survivor[update] = xtmp[update]
                r_norm_survivor[update] = r_norm_tmp[update]

            # make a full step for the momentum, except at the end of trajectory
            if j != leapfrogs - 1:
                _, gradu_prior = log_density(q, constellation_norm, mu, sigma)
                r = y - A @ q
                gradu_likelihood = - AT @ r / noise_var
                gradu = gradu_prior + gradu_likelihood
                p = p - step_size * gradu

        # make a half step for momentum at the end
        u_prior, gradu_prior = log_density(q, constellation_norm, mu, sigma)
        r = y - A @ q
        u_likelihood = np.transpose(r, axes=[0, 2, 1]) @ r / (2 * noise_var)
        gradu_likelihood = - AT @ r / noise_var
        gradu = gradu_prior + gradu_likelihood
        proposed_u = u_prior + u_likelihood
        p = p - step_size * gradu / 2
        # negate momentum at end of trajectory to make the proposal symmetric
        p = - p  # todo: why?

        # evaluate potential and kinetic energies at start and end of trajectory
        current_k = np.transpose(current_p, axes=[0, 2, 1]) @ current_p / 2
        proposed_k = np.transpose(p, axes=[0, 2, 1]) @ p / 2

        # accept or reject the state at end of trajectory
        p_uni = np.random.uniform(low=0.0, high=1.0, size=(samplers, 1, 1))
        log_pacc = np.minimum(0, current_u + current_k - proposed_u - proposed_k)
        p_acc = np.exp(log_pacc)
        index = np.squeeze(p_acc >= p_uni, axis=(1, 2))
        if index.any():
            xhat[index] = q[index]

    mse = np.squeeze(np.mean((x - x_survivor) ** 2, axis=1))
    x_hat = x_survivor[np.argmin(r_norm_survivor), :, :].copy()

    return x_hat, mse


def log_density(q, constellation_norm, mu, sigma):
    # u = - ln p(x); gradu = \par u / \par x
    m = 2 ** (mu // 2)
    dist = q - constellation_norm  # (samplers, nt, 2^(mu/2))
    exp_mat = np.exp(- dist ** 2 / (2 * sigma ** 2))
    prior_vec = np.maximum(np.sum(exp_mat, axis=2, keepdims=True) / (np.sqrt(2 * math.pi) * sigma * m),
                           1e-100)  # (samplers, nt, 1)

    prior = np.prod(prior_vec, axis=1, keepdims=True)  # (samplers, 1, 1)
    u = - np.log(prior)  # (samplers, 1)

    gradu = np.sum(- exp_mat * dist / (sigma ** 2),
                   axis=2, keepdims=True) / (np.sqrt(2 * math.pi) * sigma * m) / prior_vec # (samplers, nt , 1)

    return u, gradu
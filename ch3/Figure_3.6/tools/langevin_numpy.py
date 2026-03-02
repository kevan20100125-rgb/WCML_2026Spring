
import numpy as np
import numpy.linalg as la


def langevin(config, x, H, y, noise_var, lang):
    """

    Args:
        config: instance of SysIn
        noise_var: variance of the measurement noise
    Returns:

    """
    # SVD
    u, s, vh = la.svd(H, full_matrices=False)  # U: (..., M, K) and Vh: (..., K, N), where K = min(M, N)
    uh = u.T
    # sigma = np.zeros((2 * config.NT, 2 * config.NT))
    sigma = np.diag(s)

    # create variables to save each trajectory
    dist = np.zeros(config.n_traj)
    list_traj = np.zeros((2 * config.Nt, config.n_traj))
    mse = np.zeros(config.n_traj)
    # x_hat = np.zeros((2 * config.NT))

    # Langevin detector
    for j in range(config.n_traj):
        # run langevin
        sample_last, samples = lang.forward(s, sigma, uh, vh, y, np.sqrt(noise_var), config.Nt, config.temp)
        # Generate n_traj realizations of Langevin and then choose the best one w.r.t to ||y-Hx||^2
        list_traj[:, j] = sample_last.reshape(-1)
        dist[j] = la.norm(y - H @ sample_last)
        mse[j] = np.mean(abs(sample_last - x) ** 2)

    # pick the best trajectory
    x_hat = list_traj[:, np.argmin(dist)].reshape(-1, 1)

    return x_hat, mse


class LangevinNumpy():
    def __init__(self, sigma_annealed_noise, n_samples, step, mu):
        super(LangevinNumpy, self).__init__()
        self.num_noise_levels = sigma_annealed_noise.shape[0]
        self.n_samples = n_samples
        self.step = step
        self.sigma_gaussian = sigma_annealed_noise
        self.langevin_base = UnjustedLangevin(self.n_samples, self.step, mu)

    def forward(self, singulars, sigma, uh, vh, y, noise_sigma, nt, temp):
        r1, r2 = 1, -1
        z_init = np.random.uniform(r2, r1, (2 * nt, 1))
        zi = None
        sample_list = []
        for index in range(self.num_noise_levels):
            zi, samples = self.langevin_base.forward(z_init, singulars.reshape(-1, 1), sigma, uh, vh, y, noise_sigma,
                                                     self.sigma_gaussian[index], self.sigma_gaussian[-1], nt, temp)
            sample_list.append(samples)
            z_init = zi.copy()
        return zi, sample_list


class UnjustedLangevin():
    """
    ULA class to define the langevin algorithm plain for each level
    """
    def __init__(self, n_samples, step, mu):
        super(UnjustedLangevin, self).__init__()
        self.n_samples = n_samples
        self.step = step
        self.mu = mu
        self.mod_n = 2 ** mu
        self.constellation = self.QAM_N_const()
        self.real_QAM_const, self.imag_QAM_const = self.QAM_const()

    def forward(self, zi, singulars, sigma, uh, vh, y, noise_sigma, sigma_gaussian, sigma_l, nt, temp):
        """
        Forward pass
        Input:
            z0: initial value
            singulars: vector with singular values
            sigma: Matrix with the singular values
            y: observations
            uh, vh: left and right singular vectors
            noise_sigma: sqrt of the variance of the measurement noise
            sigma_gaussian:sqrt of the variance of the annealed noise at the level
            sigma_l: sqrt of the variance of the annealed noise at the last level
            batch_size: number of channel samples
            nt: Number of users
            M: order of the modulation (sqrt)
            temp: Temperature parameter
        Output:
            zi: estimation after the n_samples iterations
            samples: all the samples in the level
        """
        samples = []
        yt, zt = uh @ y, vh @ zi
        lam = np.zeros((2 * nt, 1))
        # define the index values corresponding to noise_sigma > or < singular * sigma_annealed
        index = noise_sigma * np.ones((2 * nt, 1)) < singulars * sigma_gaussian
        # position dependent step size
        lam[index == True] = sigma_gaussian ** 2 - noise_sigma ** 2 / singulars[index == True] ** 2
        lam[index == False] = sigma_gaussian ** 2 * (1 - singulars[index == False] ** 2 *
                                                     (sigma_gaussian / noise_sigma) ** 2)

        for i in range(self.n_samples):  # each level has n_samples
            grad = np.zeros((2 * nt, 1))
            # score of the prior
            prior = (self.gaussian(zi, sigma_gaussian ** 2, nt) - zi) / sigma_gaussian ** 2  # eq(10)
            priorMul = vh @ prior

            # score of the likelihood
            diff = yt - sigma @ zt
            cov_diag = noise_sigma ** 2 * np.ones((2 * nt, 1)) - sigma_gaussian ** 2 * singulars ** 2
            cov_diag[index == True] = -1 * cov_diag[index == True]
            cov_inv = np.diag(1 / cov_diag.reshape(-1))
            aux = cov_inv @ diff
            likelihood = sigma.T @ aux

            # score of the posterior
            if index.all() == False:
                grad[index == False] = likelihood[index == False] + priorMul[index == False]
            if index.any() == True:
                grad[index == True] = likelihood[index == True]

            # noise definition
            noiset = np.random.randn(2 * nt, 1)
            zt = zt + (self.step / sigma_l ** 2) * lam * grad +\
                 np.sqrt(2 * temp * self.step / sigma_l ** 2 * lam) * noiset  # todo
            zi = vh.T @ zt
            samples.append(zi)

        return zi, samples

    def gaussian(self, zi, sigma_annealed, nt):
        """
        Gaussian denoiser
        Returns:

        """
        # calculate the distance of the estimated symbol to each true symbol from the constellation
        arg_r = zi[0:nt] - self.real_QAM_const
        arg_i = zi[nt:] - self.imag_QAM_const

        # softmax to calculate probabilities
        zt = arg_r ** 2 + arg_i ** 2  # shape(nt, 2**mu)
        exp = np.exp(-1.0 * zt / (2.0 * sigma_annealed))
        exp = exp / np.sum(exp, axis=1, keepdims=True)

        # multiplication of the numerator with each symbol
        xr = exp * self.real_QAM_const  # shape(nt, 2**mu)
        xi = exp * self.imag_QAM_const

        # summation and concatenation to obtain the real version of the complex symbol
        xr = np.sum(xr, axis=1, keepdims=True)  # shape(nt, 1)
        xi = np.sum(xi, axis=1, keepdims=True)
        x_out = np.concatenate((xr, xi))  # shape(2*nt, 1)

        return x_out

    def QAM_const(self):
        sqrt_mod_n = np.int(np.sqrt(self.mod_n))
        real_qam_consts = np.zeros(self.mod_n, dtype=int)
        imag_qam_consts = np.zeros(self.mod_n, dtype=int)
        for i in range(sqrt_mod_n):
            for j in range(sqrt_mod_n):
                index = sqrt_mod_n * i + j
                real_qam_consts[index] = i
                imag_qam_consts[index] = j

        return(self.modulate(real_qam_consts), self.modulate(imag_qam_consts))

    # Method to get the Constellation symbols
    def QAM_N_const(self):
        n = self.mod_n
        constellation = np.linspace(int(-np.sqrt(n) + 1), int(np.sqrt(n) - 1), int(np.sqrt(n)))
        alpha = np.sqrt((constellation ** 2).mean())
        constellation /= (alpha * np.sqrt(2))
        return constellation

    def modulate(self, indices):
        x = self.constellation[indices]
        return x


def langevin_para(config, x, H, y, noise_var, lang):
    # SVD
    u, s, vh = la.svd(H, full_matrices=False)  # U: (..., M, K) and Vh: (..., K, N), where K = min(M, N)
    uh = u.T
    # sigma = np.zeros((2 * config.NT, 2 * config.NT))
    sigma = np.diag(s)

    # run langevin with n_traj realizations
    list_traj, samples = lang.forward(s, sigma, uh, vh, y, np.sqrt(noise_var), config.Nt, config.temp)
    # choose the best one w.r.t to ||y-Hx||^2
    dist = la.norm(y - H @ list_traj, axis=1, keepdims=True)
    mse = np.squeeze(np.mean(abs(list_traj - x) ** 2, axis=1))

    # pick the best trajectory
    x_hat = list_traj[np.argmin(dist), :, :].reshape(-1, 1)

    return x_hat, mse


class LangevinNumpyPara():
    def __init__(self, sigma_annealed_noise, n_samples, step, mu, n_traj):
        super(LangevinNumpyPara, self).__init__()
        self.num_noise_levels = sigma_annealed_noise.shape[0]
        self.n_samples = n_samples
        self.step = step
        self.sigma_gaussian = sigma_annealed_noise
        self.n_traj = n_traj
        self.langevin_base = UnjustedLangevinPara(self.n_samples, self.step, mu, self.n_traj)

    def forward(self, singulars, sigma, uh, vh, y, noise_sigma, nt, temp):
        r1, r2 = 1, -1
        z_init = np.random.uniform(r2, r1, (self.n_traj, 2 * nt, 1))
        zi = None
        sample_list = []
        for index in range(self.num_noise_levels):
            zi, samples = self.langevin_base.forward(z_init, singulars.reshape(-1, 1), sigma, uh, vh, y, noise_sigma,
                                                     self.sigma_gaussian[index], self.sigma_gaussian[-1], nt, temp)
            sample_list.append(samples)
            z_init = zi.copy()
        return zi, sample_list  # (n_traj, 2*nt, 1) & [num_noise_levels, n_samples, n_traj, 2*nt, 1]


class UnjustedLangevinPara(UnjustedLangevin):
    """
    ULA class to define the langevin algorithm plain for each level (parallel realization)
    """
    def __init__(self, n_samples, step, mu, n_traj):
        super(UnjustedLangevinPara, self).__init__(n_samples, step, mu)
        self.n_traj = n_traj

    def forward(self, zi, singulars, sigma, uh, vh, y, noise_sigma, sigma_gaussian, sigma_l, nt, temp):
        """
        Forward pass
        Input:
            z0: initial value
            singulars: vector with singular values
            sigma: Matrix with the singular values
            y: observations
            uh, vh: left and right singular vectors
            noise_sigma: sqrt of the variance of the measurement noise
            sigma_gaussian:sqrt of the variance of the annealed noise at the level
            sigma_l: sqrt of the variance of the annealed noise at the last level
            batch_size: number of channel samples
            nt: Number of users
            M: order of the modulation (sqrt)
            temp: Temperature parameter
        Output:
            zi: estimation after the n_samples iterations
            samples: all the samples in the level
        """
        samples = []
        yt, zt = uh @ y, vh @ zi  # (2 * nt, 1) & (ntraj, 2 * nt, 1)
        lam = np.zeros((2 * nt, 1))
        # define the index values corresponding to noise_sigma > or < singular * sigma_annealed
        index = noise_sigma * np.ones((2 * nt, 1)) < singulars * sigma_gaussian
        # position dependent step size
        lam[index == True] = sigma_gaussian ** 2 - noise_sigma ** 2 / singulars[index == True] ** 2
        lam[index == False] = sigma_gaussian ** 2 * (1 - singulars[index == False] ** 2 *
                                                     (sigma_gaussian / noise_sigma) ** 2)  # (2*nt, 1)

        for i in range(self.n_samples):  # each level has n_samples
            grad = np.zeros((self.n_traj, 2 * nt, 1))
            # score of the prior
            prior = (self.gaussian(zi, sigma_gaussian ** 2, nt) - zi) / sigma_gaussian ** 2  # eq(10)  (n_traj, 2*nt, 1)
            priorMul = vh @ prior  # (n_traj, 2*nt, 1)

            # score of the likelihood
            diff = yt - sigma @ zt  # (n_traj, 2*nt, 1)
            cov_diag = noise_sigma ** 2 * np.ones((2 * nt, 1)) - sigma_gaussian ** 2 * singulars ** 2
            cov_diag[index == True] = -1 * cov_diag[index == True]
            cov_inv = np.diag(1 / cov_diag.reshape(-1))  # (2*nt, 1)
            aux = cov_inv @ diff  # (n_traj, 2*nt, 1)
            likelihood = sigma.T @ aux  # (n_traj, 2*nt, 1)

            # score of the posterior
            if index.all() == False:
                grad[:, index == False] = likelihood[:, index == False] + priorMul[:, index == False]
            if index.any() == True:
                grad[:, index == True] = likelihood[:, index == True]

            # noise definition
            noiset = np.random.randn(self.n_traj, 2 * nt, 1)
            zt = zt + (self.step / sigma_l ** 2) * lam * grad +\
                 np.sqrt(2 * temp * self.step / sigma_l ** 2 * lam) * noiset  # (n_traj, 2*nt, 1)
            zi = vh.T @ zt  # (n_traj, 2*nt, 1)
            samples.append(zi)

        return zi, samples

    def gaussian(self, zi, sigma_annealed, nt):
        """
        Gaussian denoiser
        Returns:

        """
        # calculate the distance of the estimated symbol to each true symbol from the constellation
        arg_r = zi[:, 0:nt] - self.real_QAM_const  # (n_traj, nt, 2**mu)
        arg_i = zi[:, nt:] - self.imag_QAM_const  # (n_traj, nt, 2**mu)

        # softmax to calculate probabilities
        zt = arg_r ** 2 + arg_i ** 2  # (n_traj, nt, 2**mu)
        exp = np.exp(-1.0 * zt / (2.0 * sigma_annealed))
        exp = exp / np.sum(exp, axis=2, keepdims=True)  # (n_traj, nt, 2**mu)

        # multiplication of the numerator with each symbol
        xr = exp * self.real_QAM_const  # (n_traj, nt, 2**mu)
        xi = exp * self.imag_QAM_const  # (n_traj, nt, 2**mu)

        # summation and concatenation to obtain the real version of the complex symbol
        xr = np.sum(xr, axis=2, keepdims=True)  # (n_traj, nt, 1)
        xi = np.sum(xi, axis=2, keepdims=True)  # (n_traj, nt, 1)
        x_out = np.concatenate((xr, xi), axis=1)  # (n_traj, 2*nt, 1)

        return x_out
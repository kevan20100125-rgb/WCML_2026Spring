import numpy as np
import matplotlib.pyplot as plt
import random

np.random.seed(42)
random.seed(42)

# Define system parameters
Nt = 8  # Number of receive antennas
Nr = 8  # Number of transmit antennas
K = 2 * Nt  # Real-valued Tx
N = 2 * Nr  # Real-valued Rx
snr_db_list = np.arange(0, 30, 5)  # SNR from 0 to 25 dB with 5 dB increment
mod_type = 'QPSK'
num_symbols = 10000  # Number of symbols to simulate (each symbol represents 2 bits)
iter_num = 10  # Maximum number of OAMP iterations


# Generate Rayleigh fading channel (complex Gaussian)
def generate_real_channel():
    H_complex = (np.random.randn(Nr, Nt) + 1j * np.random.randn(Nr, Nt)) / np.sqrt(2)
    for col in range(Nt):
        norm_factor = np.linalg.norm(H_complex[:, col])
        H_complex[:, col] /= norm_factor

    # real-valued channel
    H_real = np.block([[np.real(H_complex), -np.imag(H_complex)],
                       [np.imag(H_complex), np.real(H_complex)]])
    return H_real


# OAMP Algorithm
def oamp_detector(y, H, sigma_w_sq, max_iter=10):
    """
        OAMP algorithm for MIMO detection.
        Performs iterative estimation using LMMSE and MMSE denoising.
    """
    K = H.shape[1]
    # Initialization
    x_hat = np.zeros(K)
    v_sq = 0.5

    for t in range(max_iter):
        # Step 1: Compute LMMSE estimation matrix
        # W_LMMSE = ...

        # Step 2: Normalize W_LMMSE using trace
        # tr_val = ...
        # W_t = ...

        # Step 3: Compute residual and linear estimate
        # residual = ...
        # r_t = ...

        # Step 4: Estimate posterior variance
        #tau_sq = ...

        # Nonlinear MMSE denoising step, symbol-wise MMSE for QPSK
        a0 = -1 / np.sqrt(2)
        a1 = 1 / np.sqrt(2)
        x_hat_next = np.zeros(K)

        for j in range(K):
            exp0 = -0.5 * (r_t[j] - a0) ** 2 / tau_sq
            exp1 = -0.5 * (r_t[j] - a1) ** 2 / tau_sq
            max_exp = max(exp0, exp1)
            exp_vals = np.array([exp0 - max_exp, exp1 - max_exp])
            probs = np.exp(exp_vals)
            probs /= np.sum(probs)
            x_hat_next[j] = a0 * probs[0] + a1 * probs[1]

        # Step 5: Update signal variance estimate v^2
        # residual_next = ...
        # norm_residual_sq = ...
        # v_next_sq = ...
        # v_next_sq = max(v_next_sq, 1e-9)

        x_hat = x_hat_next
        v_sq = v_next_sq
    return x_hat


# BER calculation
def simulate_oamp_ber():
    ber_results = []

    for snr_db in snr_db_list:
        # noise variance
        snr_lin = 10 ** (snr_db / 10)
        sigma_w_tilde_sq = 1.0 / snr_lin
        sigma_w_sq = sigma_w_tilde_sq / 2

        bit_errors = 0
        total_bits = 0

        for _ in range(num_symbols):
            # QPSK
            bits = np.random.randint(0, 2, 2 * Nt)
            s_complex = (2 * bits[::2] - 1 + 1j * (2 * bits[1::2] - 1)) / np.sqrt(2)
            x_real = np.concatenate([np.real(s_complex), np.imag(s_complex)])

            H_real = generate_real_channel()
            w = np.sqrt(sigma_w_sq) * np.random.randn(N)
            y = H_real @ x_real + w

            x_hat = oamp_detector(y, H_real, sigma_w_sq, max_iter=iter_num)
            bits_est = (x_hat > 0).astype(int)

            # Reorder original bits for comparison
            true_bits_ordered = np.zeros(2 * Nt, dtype=int)
            for i in range(Nt):
                true_bits_ordered[i] = bits[2 * i]
                true_bits_ordered[Nt + i] = bits[2 * i + 1]

            errors = np.sum(true_bits_ordered != bits_est)
            bit_errors += errors
            total_bits += 2 * Nt

        ber = bit_errors / total_bits
        ber_results.append(ber)
        print(f"SNR = {snr_db} dB, BER = {ber:.6f}")

    return ber_results


if __name__ == "__main__":
    ber_results = simulate_oamp_ber()

    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_db_list, ber_results, 'o-', linewidth=2)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('OAMP Detection Performance in 8x8 MIMO with QPSK')
    plt.grid(True, which='both', linestyle='--')
    plt.ylim(1e-5, 1)
    plt.xlim(0, 25)
    plt.show()
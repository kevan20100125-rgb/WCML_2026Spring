import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time

Nt = 8
Nr = 8
iter_num = 10
batch_size = 256
epochs = 50
train_size = 10000
valid_size = 2000
snr_db_list = np.arange(0, 30, 5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def generate_real_channel(batch_size):
    # Rayleigh fading
    H_complex = (torch.randn(batch_size, Nr, Nt, dtype=torch.float32) +
                 1j * torch.randn(batch_size, Nr, Nt, dtype=torch.float32)) / np.sqrt(2)

    H_real = torch.zeros(batch_size, 2 * Nr, 2 * Nt, dtype=torch.float32)
    H_real_part = torch.real(H_complex)
    H_imag_part = torch.imag(H_complex)

    H_real[:, :Nr, :Nt] = H_real_part
    H_real[:, :Nr, Nt:2 * Nt] = -H_imag_part
    H_real[:, Nr:2 * Nr, :Nt] = H_imag_part
    H_real[:, Nr:2 * Nr, Nt:2 * Nt] = H_real_part
    return H_real.to(device)

def generate_data(batch_size, snr_db):
    snr_lin = 10 ** (snr_db / 10)
    sigma_w_sq = 1.0 / snr_lin / 2  # noise variance

    # QPSK modulation
    bits = torch.randint(0, 2, (batch_size, 2 * Nt), dtype=torch.float32)
    x_real = torch.zeros(batch_size, 2 * Nt, dtype=torch.float32)
    real_bits = bits[:, :Nt]
    imag_bits = bits[:, Nt:2 * Nt]
    x_real[:, :Nt] = (2 * real_bits - 1) / np.sqrt(2)
    x_real[:, Nt:2 * Nt] = (2 * imag_bits - 1) / np.sqrt(2)

    # y, H, x
    H_real = generate_real_channel(batch_size)
    w = torch.sqrt(torch.tensor(sigma_w_sq)) * torch.randn(batch_size, 2 * Nr, dtype=torch.float32)
    y = torch.bmm(H_real, x_real.unsqueeze(-1)).squeeze(-1) + w
    return y.to(device), H_real.to(device), x_real.to(device)

# OAMP-Net Task c: gamma_t, theta_t, phi_t, xi_t
class OAMPNet_c(nn.Module):
    def __init__(self, iter_num):
        super(OAMPNet_c, self).__init__()
        self.iter_num = iter_num
        self.gamma = nn.Parameter(torch.ones(iter_num, dtype=torch.float32))
        self.theta = nn.Parameter(torch.ones(iter_num, dtype=torch.float32))
        self.phi = nn.Parameter(torch.ones(iter_num, dtype=torch.float32))
        self.xi = nn.Parameter(torch.zeros(iter_num, dtype=torch.float32))

    def forward(self, y, H, sigma_w_sq):
        batch_size = y.shape[0]
        K = 2 * Nt
        N = 2 * Nr
        x_hat = torch.zeros(batch_size, K, dtype=torch.float32, device=device)
        v_sq = 0.5

        for t in range(self.iter_num):
            # Step 1: Compute LMMSE matrix
            # H_trans = ...
            # H_Ht = ...
            # reg = ...
            # mat = ...
            # inv_mat = ...
            # W_LMMSE = ...

            # Step 2: Compute scaled W_t using trace normalization
            # W_H = ...
            # tr_val = ...
            # tr_val = ...
            # scale =  ...
            # W_t =  ...

            # Setp 3: Compute linear residual r_t using learnable γ_t
            # Hx = ...
            # residual = ...
            # W_res = ...
            # r_t = ...

            # Estimate the denoiser variance tau_sq with theta_t
            B_t = torch.eye(K, device=device).unsqueeze(0) - self.theta[t] * torch.bmm(W_t, H)
            B_Bt = torch.bmm(B_t, B_t.transpose(1, 2))
            term1 = (1 / K) * torch.diagonal(B_Bt, dim1=1, dim2=2).sum(dim=1) * v_sq
            W_Wt = torch.bmm(W_t, W_t.transpose(1, 2))
            term2 = (1 / (2 * K)) * torch.diagonal(W_Wt, dim1=1, dim2=2).sum(dim=1) * sigma_w_sq
            tau_sq = term1 + term2

            # Step 4: Nonlinear MMSE denoising step with learnalbe phi_t, xi_t
            # x_mmse = ...
            # x_hat_next = ...

            # Step 5: Update v^2 based on next residual
            # Hx_next = ...
            # residual_next = ...
            # norm_residual_sq = ...
            # HtH = ...
            # trace_HtH = ...
            # v_next_sq = ...

            # Update for next iteration
            x_hat = x_hat_next
            v_sq = v_next_sq.mean().item()

        return x_hat

    def mmse_denoiser(self, r_t, tau_sq):
        """
            Symbol-wise MMSE denoising for real-valued QPSK.
        """
        a0 = -1 / np.sqrt(2)
        a1 = 1 / np.sqrt(2)
        x_hat_next = torch.zeros_like(r_t)

        for j in range(r_t.shape[1]):
            r_j = r_t[:, j]
            exp0 = -0.5 * (r_j - a0) ** 2 / tau_sq
            exp1 = -0.5 * (r_j - a1) ** 2 / tau_sq
            max_exp = torch.max(exp0, exp1)
            exp0_norm = exp0 - max_exp
            exp1_norm = exp1 - max_exp
            exp_sum = torch.exp(exp0_norm) + torch.exp(exp1_norm)
            prob0 = torch.exp(exp0_norm) / exp_sum
            prob1 = torch.exp(exp1_norm) / exp_sum
            x_hat_next[:, j] = a0 * prob0 + a1 * prob1

        return x_hat_next


# train
def train_model(model, snr_db):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
    criterion = nn.MSELoss()  # MSE loss

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        start_time = time.time()

        y_train, H_train, x_train = generate_data(train_size, snr_db)

        dataset = torch.utils.data.TensorDataset(y_train, H_train, x_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for batch_idx, (y_batch, H_batch, x_batch) in enumerate(loader):
            optimizer.zero_grad()
            x_pred = model(y_batch, H_batch, 1.0 / (10 ** (snr_db / 10)) / 2)
            loss = criterion(x_pred, x_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}, Time: {epoch_time:.2f}s")
    return model


# test
def test_model_ber(model, snr_db):
    model.eval()
    num_symbols = 10000  # Total number of test symbols
    sigma_w_sq = 1.0 / (10 ** (snr_db / 10)) / 2  # Noise variance

    bit_errors = 0
    total_bits = 0

    with torch.no_grad():
        for i in range(0, num_symbols, batch_size):
            batch_size_current = min(batch_size, num_symbols - i)
            y_batch, H_batch, x_batch = generate_data(batch_size_current, snr_db)  # Generate new test data
            x_pred = model(y_batch, H_batch, sigma_w_sq)  # Predict symbols
            bits_pred = (x_pred > 0).int()
            bits_true = (x_batch > 0).int()

            errors = torch.sum(bits_true != bits_pred)  # bit errors
            bit_errors += errors.item()
            total_bits += batch_size_current * 2 * Nt

    ber = bit_errors / total_bits
    print(f"SNR = {snr_db} dB, BER = {ber:.6f}")
    return ber


def main():
    print("\nTraining OAMP-Net (task c)...")
    model_c = OAMPNet_c(iter_num)
    model_c = train_model(model_c, snr_db=20)  # train SNR = 20 dB

    ber_results_c = []

    print("\nTesting OAMP-Net (task c)...")
    for snr_db in snr_db_list:
        ber = test_model_ber(model_c, snr_db)
        ber_results_c.append(ber)

    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_db_list, ber_results_c, 's-', label='OAMP-Net-c')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('MIMO Detection Performance in 8x8 System with QPSK')
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plt.ylim(1e-5, 1)
    plt.xlim(0, 25)
    plt.show()


if __name__ == "__main__":
    main()
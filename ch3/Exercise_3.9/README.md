# Exercise 3.9: OAMP and OAMP-Net for MIMO Detection

This repository provides the skeleton code for Exercise 3.9. Your task is to implement the **OAMP** (Orthogonal Approximate Message Passing) algorithm and its deep learning-unrolled version, **OAMP-Net**, to solve the MIMO signal detection problem. You will compare the performance of the traditional iterative algorithm with the learning-based approaches.

## Experiment Setup
The scripts are pre-configured with the following system parameters:
* **Antenna Configuration:** $8 \times 8$ MIMO ($N_t = 8, N_r = 8$)
* **Modulation Scheme:** QPSK ($\mu = 2$)
* **Channel Model:** Rayleigh fading channel (Real-valued decomposition used)
* **Network/Iteration Depth:** 10 iterations (layers)
* **SNR Range:** 0 dB to 25 dB with 5 dB increment
* **Training Setup:** (For Task b & c) Supervised learning with MSE loss, Adam optimizer, SNR = 20 dB for training.

## What You Need To Do

| Checklist | Details |
| :---- | :--- |
| **(a) OAMP** | Open `OAMP_ex_a.py`. Fill in the **5 key steps** of the OAMP algorithm: (1) LMMSE matrix $W_{LMMSE}$, (2) Trace normalization for $W_t$, (3) Linear residual $r_t$, (4) Posterior variance $\tau^2$, and (5) Signal variance $v^2$ update. |
| **(b) OAMPNet-b** | Open `OAMP_ex_b.py`. Implement the OAMP-Net within a PyTorch `nn.Module`. Use learnable parameters $(\gamma_t, \theta_t)$ as defined in (3.73) and (3.74). Train the model and plot the BER curve. |
| **(c) OAMPNet-c** | Open `OAMP_ex_c.py`. Extend the model to make $(\gamma_t, \theta_t, \phi_t, \xi_t)$ all learnable parameters. The parameters $(\phi_t, \xi_t)$ are used to optimize the nonlinear MMSE denoiser. |
| **(d) Comparison** | Run all three scripts and compare the resulting BER vs. SNR curves. Observe how learning-based parameters affect the convergence and final performance. |

## How to Run
| Task | Execution Command |
| :--- | :--- |
| **Run OAMP** | `python OAMP_ex_a.py` |
| **Train/Test OAMP-Net-b** | `python OAMP_ex_b.py` |
| **Train/Test OAMP-Net-c** | `python OAMP_ex_c.py` |

> **Note:** The PyTorch scripts will automatically detect if a GPU (CUDA) is available.

## Hint
* **Mathematical Reference:** Please strictly follow the formulations in **Section 3.2.3** for standard OAMP and **Section 3.3.2** for the OAMP-Net architecture.
* **Denoiser:** In Task (c), ensure that the learnable $\phi_t$ and $\xi_t$ are properly integrated into the MMSE denoising function as described in equation (3.76).
* **Matrix Inversion:** In the LMMSE step, remember to add a small regularization term (noise variance) to the diagonal to ensure numerical stability during matrix inversion.

## Files
| File | Purpose |
| :--- | :--- |
| `OAMP_ex_a.py` | NumPy-based skeleton code for the traditional OAMP algorithm (Task a). |
| `OAMP_ex_b.py` | PyTorch-based skeleton code for OAMP-Net with learnable $(\gamma, \theta)$ (Task b). |
| `OAMP_ex_c.py` | PyTorch-based skeleton code for OAMP-Net with learnable $(\gamma, \theta, \phi, \xi)$ (Task c). |
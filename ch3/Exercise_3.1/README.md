# Exercise 3.1: Learning-based Signal Detection for OFDM Systems

This repository provides the starter code for Exercise 3.1. Your task is to use Deep Learning (FC-DNN) to implicitly estimate the channel and recover transmitted bits in an OFDM system, based on the paper by Ye et al. [15]. You will modify network dimensions, modulation schemes, and pilot configurations to evaluate the system's Bit Error Rate (BER).


## What You Need to Do

| Checklist | Details |
|-----------|---------|
| **Task (b)** | Open `main.py`. Implement a loop to iterate `config.SNR` from 5 to 25 dB. <br> Modify `config.Pilots` (e.g., 8, 16, 64) to evaluate the impact of **different pilot numbers**, and simulate a **no-pilot** scenario. Collect BER results to reproduce Figure 3.3. |
| **Task (c)** | Open `Train.py` and `Test.py` and change `mu = 6` for **64-QAM**. <br> Open `main.py` and change `config.pred_range = np.arange(48, 96)`. <br> Open `Train.py` and change the network output size to `n_output = 48`. |
| **Task (d)** | Revert to QPSK (`mu = 2`). Implement a **single large DNN**: <br> • Open `main.py` and set `config.pred_range = np.arange(0, 128)`. <br> • Open `Train.py` and set `n_output = 128`. <br> • Increase `n_hidden_1`, `n_hidden_2`, etc., in `Train.py` to give the network enough capacity. |
| **Run** | Execute: `python main.py` for each specific configuration. |
| **Observe** | The console will output the testing BER. You should manually record these values to plot BER vs. SNR curves and compare the performance differences in your report. |


> **Hint:** The codes have been tested on Ubuntu 16.04 + tensorflow 1.1 + Python 2.7.



## Files

| File | Purpose |
|------|---------|
| `main.py` | Main script where you configure hyperparameters (`sysconfig`) and run the pipeline. |
| `DNN_Detection/Train.py` | Contains the FC-DNN architecture definition and the training loop. |
| `DNN_Detection/Test.py` | Evaluates the trained models and calculates the final BER. |
| `DNN_Detection/utils.py` | Mathematical formulations for OFDM (IDFT, CP addition, channel convolution). |
| `H_dataset/` | Pre-generated dataset containing Rayleigh fading channel responses. |
> **Hint:** The H_dataset can be downloaded from the following link: https://github.com/haoyye/OFDM_DNN.

[15] H. Ye, G. Y. Li, and B.-H. Juang, “Power of deep learning for channel estimation and signal detection in OFDM systems,” *IEEE Wireless Communications Letters*,
vol. 7, no. 1, pp. 114–117, Feb. 2018.
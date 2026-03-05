# Exercise 3.10: EP and EPNet for MIMO Detection
This repository provides the starter code for Exercise 3.10, which focuses on model-based signal detection algorithms for MIMO systems. The goal of this exercise is to implement the EP detector and its deep-unfolded learning-enhanced version (EPNet), and to evaluate their BER performance in a Rayleigh fading MIMO environment.

## Experiment Setup

The script is pre-configured with the specific parameters from the textbook:

* **MIMO System:** 8 × 8 MIMO


* **Modulation Scheme:** QPSK ($\mu = 2$)

* **Channel Model:** Rayleigh fading channel

* **Detection Algorithms:** EP and EPNet

* **Number of Iterations / Layers $T$ :** 4

* **Conjugate Gradient Iterations $i_{\text{cg}}$ :** 50

* **Damping Factors $\beta$ :**
  * Fixed for EP
  * Learnable (layer-wise) for EPNet

* **Loss Function:** Supervised loss defined in Eq. (3.77)

* **SNR Range:** 0 dB to 25 dB (step size: 5 dB)


## What You Need to Do

| Checklist | Details |
|-----------|---------|
| **Select Detector** | Open `main.py` and choose the detection algorithm by setting `detect_type` to either `'EP'` or `'EPNet'`.  |
| **Implement EP** | Open `tools/EP.py` and locate the block marked with **`# YOUR CODE HERE`**. Complete the implementation of the EP algorithm for MIMO signal detection.
| **Implement EPNet** | Open `tools/networks.py` and locate the function **`build_EP(trainSet)`**, where the block marked with **`# YOUR CODE HERE`** is provided. Replace the placeholder with the deep-unfolded EPNet implementation.
| **Run** | Running `main.py` will generate the BER results under the selected detector. |
> **Hint:**  
> When implementing EPNet, use a sigmoid function to constrain the damping factors $\beta$ to the range $(0,1)$. Defining $\beta$ as a `tf.Variable` enables end-to-end training via backpropagation and improves convergence stability.




## Files

| File | Purpose |
|------|---------|
| `main.py` | Main entry script. Selects the detection algorithm (`EP` or `EPNet`) and launches BER simulations. |
| `tools/EP.py` | Implementation of the conventional EP detector. Contains a `# YOUR CODE HERE` block for Exercise 3.10(a). |
| `tools/networks.py` | Implementation of the deep-unfolded detector. The function `build_EP(trainSet)` contains a `# YOUR CODE HERE` block for Exercise 3.10(b). |
| `tools/utils.py` | Signal processing utilities. Implements QPSK/16QAM/64QAM modulation and demodulation, OFDM processing , channel modeling, nonlinear estimator, and LMMSE channel estimation. |
| `tools/MIMO_detection.py` | Core end-to-end MIMO simulation script. Handles bit generation, modulation, channel transmission, detector invocation, and BER/MSE evaluation. |
| `tools/problems.py` | Defines the MIMO detection problem, including system dimensions, channel model, and TensorFlow placeholders. |

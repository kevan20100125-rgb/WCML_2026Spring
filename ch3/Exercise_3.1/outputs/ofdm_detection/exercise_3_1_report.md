# Exercise 3.1 Report Draft

## Exercise 3.1(a): FC-DNN input and output dimensions for 64-QAM

The OFDM system uses 64 subcarriers. The FC-DNN input consists of two received OFDM blocks: the pilot block and the data block. Each OFDM block has 64 complex-valued frequency-domain samples. After separating real and imaginary parts, one block contributes

```text
64 complex samples x 2 real components = 128 real numbers.
```

Because the DNN input concatenates the received pilot block and the received data block, the input layer size is

```text
128 + 128 = 256.
```

For 64-QAM, each subcarrier carries

```text
log2(64) = 6 bits.
```

Therefore, one 64-subcarrier OFDM data symbol contains

```text
64 x 6 = 384 bits.
```

If the receiver uses 8 identical FC-DNNs, each DNN predicts

```text
384 / 8 = 48 bits.
```

Thus, the per-DNN architecture for task (c) is

```text
256 -> 500 -> 250 -> 120 -> 48.
```

Only simulating one of the eight FC-DNNs is not sufficient for a complete BER result. One model only covers 48 of the 384 transmitted bits in a 64-QAM OFDM data symbol. A valid full BER must concatenate the outputs of all eight DNNs and compute bit errors over all 384 transmitted bits.

## Exercise 3.1(b): Reproduction of the QPSK BER experiment

The implemented task (b) uses QPSK, 64 subcarriers, CP length 16, SNR values 5, 10, 15, 20, and 25 dB, and pilot counts 8 and 64. It compares 8 FC-DNNs with LS and LMMSE baselines. Each FC-DNN input has dimension 256 and each QPSK DNN output group has 16 bits. The eight output groups are concatenated to form the full 128-bit prediction.

### Current task (b) BER results

| Curve | 5 dB | 10 dB | 15 dB | 20 dB | 25 dB |
|---|---:|---:|---:|---:|---:|
| LS, P=64 | 0.144534 | 0.053902 | 0.018162 | 0.005767 | 0.001836 |
| LS, P=8 | 0.165592 | 0.094296 | 0.065632 | 0.054164 | 0.053039 |
| LMMSE, P=64 | 0.088121 | 0.031252 | 0.010488 | 0.003335 | 0.001098 |
| LMMSE, P=8 | 0.126919 | 0.052386 | 0.020631 | 0.008958 | 0.005479 |
| 8 FC-DNNs, P=64 | 0.136677 | 0.061532 | 0.030110 | 0.017588 | 0.014029 |
| 8 FC-DNNs, P=8 | 0.189425 | 0.088439 | 0.046327 | 0.031716 | 0.027628 |

The result shows that FC-DNN BER decreases as SNR increases and improves when the number of pilots increases. This confirms that the DNN learns useful structure from the OFDM received samples. However, the current result does not fully reproduce the reference Fig. 3.3 trend. At 25 dB and 64 pilots, the FC-DNN BER is 0.014029, while LMMSE BER is 0.001098. At 25 dB and 8 pilots, the FC-DNN BER is 0.027628, while LMMSE BER is 0.005479.

Therefore, the correct interpretation is that the implementation is a partial reproduction. It reproduces the basic SNR and pilot-count behavior, but it does not reproduce the paper's stronger claim that the DNN is comparable to MMSE or more robust than MMSE under limited pilots. Possible reasons include channel-dataset mismatch, a stronger LMMSE covariance estimate, limited DNN capacity, and training-protocol differences.

## Exercise 3.1(c): Changing the modulation from QPSK to 64-QAM

For 64-QAM, each subcarrier carries 6 bits instead of 2 bits. The total number of bits per OFDM data symbol increases from 128 in QPSK to 384 in 64-QAM. With 8 FC-DNNs, each DNN output group increases from 16 bits to 48 bits.

### Current task (c) BER results

| Curve | 5 dB | 10 dB | 15 dB | 20 dB | 25 dB |
|---|---:|---:|---:|---:|---:|
| LS, P=64 | 0.339466 | 0.244548 | 0.150239 | 0.074255 | 0.030130 |
| LS, P=8 | 0.366369 | 0.295700 | 0.234094 | 0.192671 | 0.171206 |
| LMMSE, P=64 | 0.292377 | 0.195414 | 0.107714 | 0.045993 | 0.016404 |
| LMMSE, P=8 | 0.350396 | 0.263627 | 0.177831 | 0.107611 | 0.063438 |
| 8 FC-DNNs, P=64 | 0.325489 | 0.222178 | 0.131434 | 0.068130 | 0.032533 |
| 8 FC-DNNs, P=8 | 0.369946 | 0.290695 | 0.206233 | 0.143324 | 0.108196 |

The 64-QAM BER is higher than the QPSK BER. This is expected because 64-QAM has denser constellation points and therefore smaller decision margins under the same SNR. For example, with 64 pilots at 25 dB, FC-DNN BER increases from 0.014029 in QPSK to 0.032533 in 64-QAM. With 8 pilots at 25 dB, FC-DNN BER increases from 0.027628 in QPSK to 0.108196 in 64-QAM.

This result shows that the higher-order modulation setting is substantially harder. The current FC-DNN remains functional, but it does not outperform LMMSE in the implemented setting.

## Exercise 3.1(d): Single FC-DNN versus eight FC-DNNs

Task (d) replaces the original eight-DNN design with one single FC-DNN that predicts the full QPSK output vector. For QPSK and 64 subcarriers, the output dimension is

```text
64 x 2 = 128 bits.
```

The eight-DNN design uses eight models with 16-bit outputs. The single-DNN design uses one larger model with 128-bit output.

### Current task (d) BER results

| Curve | 5 dB | 10 dB | 15 dB | 20 dB | 25 dB |
|---|---:|---:|---:|---:|---:|
| LS, P=64 | 0.144534 | 0.053902 | 0.018162 | 0.005767 | 0.001836 |
| LMMSE, P=64 | 0.088121 | 0.031252 | 0.010488 | 0.003335 | 0.001098 |
| 8 FC-DNNs, P=64 | 0.134403 | 0.055623 | 0.023948 | 0.012942 | 0.009579 |
| Single FC-DNN, P=64 | 0.273401 | 0.264388 | 0.203042 | 0.149519 | 0.152579 |

The eight-DNN design is clearly better in the current implementation. At 25 dB, the eight-DNN BER is 0.009579, while the single-DNN BER is 0.152579. This indicates that directly replacing eight specialized predictors with one full-output predictor makes the optimization problem much harder.

The comparison should be interpreted carefully because it is not fully parameter-matched. The QPSK eight-DNN design has approximately 2,286,448 total parameters, while the current single-DNN design has approximately 914,878 parameters. Therefore, the result supports the conclusion that the implemented single-DNN replacement is insufficient, but it does not prove that every possible single-DNN architecture is inferior.

## Overall conclusion

The current code base satisfies the main implementation requirements for tasks (b), (c), and (d), and the analytical answer for task (a) is now provided. The strongest remaining limitation is task (b): the implementation produces meaningful BER curves, but it should not be claimed as a perfect reproduction of the reference paper's Fig. 3.3. The final report should explicitly state this limitation and discuss likely causes.

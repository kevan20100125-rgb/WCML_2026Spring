# Task (b) Reproduction Note

## Purpose

Task (b) asks for a reproduction of the reference BER trend for learning-based OFDM detection. The current implementation produces complete BER curves for QPSK with 8 and 64 pilots, comparing FC-DNN, LS, and LMMSE. The result is technically useful, but it should not be described as a perfect reproduction of the reference Fig. 3.3.

## What the current result successfully shows

1. The FC-DNN curves are no longer random guessing. BER decreases as SNR increases.
2. More pilots improve FC-DNN performance. At 25 dB, FC-DNN improves from 0.027628 with 8 pilots to 0.014029 with 64 pilots.
3. LS with 8 pilots shows a high-SNR floor, which is qualitatively consistent with the limited-pilot difficulty.
4. The implemented pipeline evaluates all eight bit groups, so the reported FC-DNN BER is full-output BER rather than a one-group proxy.

## What does not match the reference trend

The reference paper reports that the learning-based detector is comparable to MMSE with enough pilots and is more robust than traditional methods when fewer pilots are used. In the current result, LMMSE remains substantially stronger than FC-DNN at high SNR.

| Condition | FC-DNN BER | LMMSE BER | Ratio |
|---|---:|---:|---:|
| QPSK, 64 pilots, 25 dB | 0.014029 | 0.001098 | 12.78x |
| QPSK, 8 pilots, 25 dB | 0.027628 | 0.005479 | 5.04x |

Therefore, the current task (b) result should be described as a partial or independent reproduction attempt.

## Likely technical causes

1. **Channel model mismatch.** The project uses the available `H_dataset` files rather than a guaranteed identical WINNER II simulation pipeline. Even if the file source is related to the original code, the exact generation statistics and sample split can affect the result.
2. **LMMSE may be stronger than the plotted reference baseline.** The current LMMSE estimator uses covariance estimated from the training channel files and applies it directly to the test channels. If this covariance is cleaner or better matched than the reference implementation, LMMSE can become a very strong baseline.
3. **FC-DNN capacity and optimization are still limited.** The task (b) DNN uses 60,000 samples per condition and 120 epochs. It learns meaningful BER curves, but the result suggests that this is not enough to match the strongest reference claim.
4. **Per-SNR model training changes the comparison protocol.** The current code trains separate models per SNR condition. This is acceptable for controlled BER curves, but it is not necessarily identical to the paper's training protocol.
5. **Feature distribution and normalization matter.** The current implementation uses checkpoint-level feature normalization. This fixed the previous random-guessing behavior, but the learned mapping may still be sensitive to pilot placement, channel statistics, and SNR-specific distributions.

## Correct wording for the report

Use wording like this:

> The reproduced QPSK experiment confirms that FC-DNN detection learns a useful OFDM input-output mapping: BER decreases with SNR and improves when the number of pilots increases. However, the current implementation does not fully match the reference Fig. 3.3 trend. In particular, LMMSE remains better than FC-DNN at high SNR for both 8-pilot and 64-pilot settings. We therefore treat the result as an independent reproduction attempt using the available channel dataset, rather than as a high-fidelity reproduction of all quantitative claims in [15].

Do not write that the current DNN is comparable to or better than LMMSE in task (b). The current CSV values do not support that claim.

## Optional next technical work

A stricter reproduction attempt should examine the exact channel generation pipeline, pilot placement, LMMSE covariance construction, training sample count, model size, and whether the reference used a different training/testing split. The highest-value next experiment is to reduce the LMMSE information advantage or exactly match the reference implementation before increasing DNN capacity.

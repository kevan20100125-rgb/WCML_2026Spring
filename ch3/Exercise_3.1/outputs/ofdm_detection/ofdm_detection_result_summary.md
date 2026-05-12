# OFDM Detection Result Summary

This file has been regenerated from the current CSV outputs under `outputs/ofdm_detection/`. It replaces the stale summary that previously showed near-random FC-DNN BER values around 0.5.

## Current result files

| File | Meaning | Status |
|---|---|---|
| `results_b.csv` | Task (b), QPSK, 8 FC-DNNs, LS, LMMSE, pilots 8 and 64 | Current |
| `results_c.csv` | Task (c), 64-QAM, 8 FC-DNNs, LS, LMMSE, pilots 8 and 64 | Current |
| `results_d.csv` | Task (d), QPSK, single FC-DNN versus 8 FC-DNNs, pilots 64 | Current |
| `run_config.json` | Last executed runner configuration only | Not a complete record for all tasks |

The `run_config.json` file only reflects the most recent task (d) run. The full b/c/d settings should be read from the CSV columns and the commands listed in `README.md`.

### Task (b): QPSK BER

| Curve | 5 dB | 10 dB | 15 dB | 20 dB | 25 dB |
|---|---:|---:|---:|---:|---:|
| LS, P=64 | 0.144534 | 0.053902 | 0.018162 | 0.005767 | 0.001836 |
| LS, P=8 | 0.165592 | 0.094296 | 0.065632 | 0.054164 | 0.053039 |
| LMMSE, P=64 | 0.088121 | 0.031252 | 0.010488 | 0.003335 | 0.001098 |
| LMMSE, P=8 | 0.126919 | 0.052386 | 0.020631 | 0.008958 | 0.005479 |
| 8 FC-DNNs, P=64 | 0.136677 | 0.061532 | 0.030110 | 0.017588 | 0.014029 |
| 8 FC-DNNs, P=8 | 0.189425 | 0.088439 | 0.046327 | 0.031716 | 0.027628 |

**Task (b) interpretation.** The FC-DNN BER decreases with SNR and with more pilots, so the DNN is learning useful structure. However, it does not reproduce the strongest claim of [15]. At 25 dB and 64 pilots, FC-DNN BER is 0.014029, while LMMSE BER is 0.001098. At 25 dB and 8 pilots, FC-DNN BER is 0.027628, while LMMSE BER is 0.005479. This should be reported as a partial reproduction rather than a perfect reproduction of Fig. 3.3.

### Task (c): 64-QAM BER

| Curve | 5 dB | 10 dB | 15 dB | 20 dB | 25 dB |
|---|---:|---:|---:|---:|---:|
| LS, P=64 | 0.339466 | 0.244548 | 0.150239 | 0.074255 | 0.030130 |
| LS, P=8 | 0.366369 | 0.295700 | 0.234094 | 0.192671 | 0.171206 |
| LMMSE, P=64 | 0.292377 | 0.195414 | 0.107714 | 0.045993 | 0.016404 |
| LMMSE, P=8 | 0.350396 | 0.263627 | 0.177831 | 0.107611 | 0.063438 |
| 8 FC-DNNs, P=64 | 0.325489 | 0.222178 | 0.131434 | 0.068130 | 0.032533 |
| 8 FC-DNNs, P=8 | 0.369946 | 0.290695 | 0.206233 | 0.143324 | 0.108196 |

**Task (c) interpretation.** 64-QAM is harder than QPSK because each subcarrier carries 6 bits and the constellation points are denser. With 64 pilots at 25 dB, FC-DNN BER increases from 0.014029 in QPSK to 0.032533 in 64-QAM. With 8 pilots at 25 dB, FC-DNN BER increases from 0.027628 in QPSK to 0.108196 in 64-QAM.

### Task (d): Single FC-DNN versus 8 FC-DNNs

| Curve | 5 dB | 10 dB | 15 dB | 20 dB | 25 dB |
|---|---:|---:|---:|---:|---:|
| LS, P=64 | 0.144534 | 0.053902 | 0.018162 | 0.005767 | 0.001836 |
| LMMSE, P=64 | 0.088121 | 0.031252 | 0.010488 | 0.003335 | 0.001098 |
| 8 FC-DNNs, P=64 | 0.134403 | 0.055623 | 0.023948 | 0.012942 | 0.009579 |
| Single FC-DNN, P=64 | 0.273401 | 0.264388 | 0.203042 | 0.149519 | 0.152579 |

**Task (d) interpretation.** The naive single FC-DNN performs much worse than the eight-DNN design. At 25 dB, the eight-DNN BER is 0.009579, while the single-DNN BER is 0.152579. This supports the conclusion that directly expanding one network to predict all bits is a harder optimization problem. The comparison is not strictly parameter-matched, so the correct conclusion is that the implemented single-DNN design is insufficient, not that every possible single-DNN design must be inferior.

## Architecture and parameter-count reference

| Case | Architecture | Parameter count |
|---|---|---:|
| QPSK, one of 8 FC-DNNs | 256-500-250-120-16 | 285,806 |
| QPSK, 8 FC-DNN total | 8 x 256-500-250-120-16 | 2,286,448 |
| QPSK, current single FC-DNN | 256-1000-500-250-128 | 914,878 |
| 64-QAM, one of 8 FC-DNNs | 256-500-250-120-48 | 289,678 |
| 64-QAM, 8 FC-DNN total | 8 x 256-500-250-120-48 | 2,317,424 |

## Final readiness checklist

| Requirement | Current status |
|---|---|
| Task (a) written answer | Added in `exercise_3_1_report.md` and `.tex` |
| Task (b) BER simulation | Complete CSV and figures, but only partial reference reproduction |
| Task (c) 64-QAM extension | Complete CSV and figures |
| Task (d) single-DNN comparison | Complete CSV and figures |
| README | Updated to current runner and result structure |
| Reproduction caveat | Added in `task_b_reproduction_note.md` |

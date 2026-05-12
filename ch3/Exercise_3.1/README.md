# Exercise 3.1: FC-DNN Detection for OFDM Systems

This repository contains the current implementation and results for Exercise 3.1. The project studies learning-based signal detection for an OFDM receiver following the setup of Ye, Li, and Juang [15]. The implemented receiver uses the received pilot OFDM block and the received data OFDM block as a 256-dimensional real-valued feature vector and predicts transmitted bits directly with fully connected DNNs.

## Current implementation status

The original starter workflow has been replaced by a reproducible PyTorch runner:

```bash
cd DNN_Detection
python run_ofdm_detection.py --task b
python run_ofdm_detection.py --task c
python run_ofdm_detection.py --task d
```

The main implementation is now organized as follows:

| Path | Role |
|---|---|
| `DNN_Detection/run_ofdm_detection.py` | Main experiment runner for tasks sanity, b, c, d, and all. |
| `DNN_Detection/ofdm_detection_core.py` | OFDM simulation, QPSK/64-QAM mapping, pilot generation, channel estimation, LS/LMMSE baselines, and BER computation. |
| `DNN_Detection/Main.py` | Thin wrapper that calls `run_ofdm_detection.py`. |
| `DNN_Detection/tools/plot_ofdm_detection_results.py` | Produces cleaner report-facing BER figures under `outputs/ofdm_detection/figures/`. |
| `outputs/ofdm_detection/results_b.csv` | Task (b) QPSK BER results. |
| `outputs/ofdm_detection/results_c.csv` | Task (c) 64-QAM BER results. |
| `outputs/ofdm_detection/results_d.csv` | Task (d) single-DNN versus eight-DNN BER results. |
| `outputs/ofdm_detection/exercise_3_1_report.md` | Report-ready written answer for Exercise 3.1. |
| `outputs/ofdm_detection/exercise_3_1_report.tex` | Overleaf-ready LaTeX section for Exercise 3.1. |
| `outputs/ofdm_detection/task_b_reproduction_note.md` | Technical note explaining the remaining gap in reproducing the reference Fig. 3.3 trend. |

## Experiment setup

| Item | Current setting |
|---|---|
| OFDM subcarriers | 64 |
| CP length | 16 |
| Feature vector | real/imag parts of received pilot block and received data block, dimension 256 |
| Training channels | `H_dataset/1.txt` to `H_dataset/300.txt` |
| Testing channels | `H_dataset/301.txt` to `H_dataset/400.txt` |
| Channel rows per file | 512 in the high-quality runs |
| Test samples | 8192 per SNR condition |
| SNR points | 5, 10, 15, 20, 25 dB |
| Traditional baselines | LS and LMMSE |
| DNN loss | `BCEWithLogitsLoss` |
| Optimizer | RMSprop |
| Feature preprocessing | per-checkpoint feature mean/std normalization |

## Commands corresponding to the current CSV results

Task (b): QPSK, 8 FC-DNNs, pilots 64 and 8.

```bash
cd DNN_Detection
python run_ofdm_detection.py   --task b   --channel-root ../H_dataset   --output-dir ../outputs/ofdm_detection   --snrs 5 10 15 20 25   --pilots 64 8   --epochs 120   --train-samples 60000   --test-samples 8192   --batch-size 512   --learning-rate 0.001   --channel-rows-per-file 512   --hidden 500 250 120   --force   --overwrite-results
```

Task (c): 64-QAM, 8 FC-DNNs, pilots 64 and 8.

```bash
cd DNN_Detection
python run_ofdm_detection.py   --task c   --channel-root ../H_dataset   --output-dir ../outputs/ofdm_detection   --snrs 5 10 15 20 25   --pilots 64 8   --epochs 180   --train-samples 120000   --test-samples 8192   --batch-size 512   --learning-rate 0.001   --channel-rows-per-file 512   --hidden 500 250 120   --force   --overwrite-results
```

Task (d): QPSK, single FC-DNN versus 8 FC-DNNs, pilots 64.

```bash
cd DNN_Detection
python run_ofdm_detection.py   --task d   --channel-root ../H_dataset   --output-dir ../outputs/ofdm_detection   --snrs 5 10 15 20 25   --pilots 64   --epochs 200   --train-samples 100000   --test-samples 8192   --batch-size 512   --learning-rate 0.001   --channel-rows-per-file 512   --hidden 500 250 120   --single-hidden 1000 500 250   --force   --overwrite-results
```

## Current assessment

The project now contains complete executable outputs for tasks (b), (c), and (d), and the report files now include the missing written answer for task (a). The main remaining technical limitation is task (b): the FC-DNN learns meaningful BER curves, but the result is not a high-fidelity reproduction of the reference Fig. 3.3 trend because LMMSE remains substantially stronger than FC-DNN at high SNR in the current implementation.

For the final report, do not claim that the current task (b) perfectly reproduces the reference paper. State that it is an independent reproduction attempt using the available `H_dataset` channel files and that the observed trend partially matches the expected SNR monotonicity but not the paper's claimed DNN robustness over MMSE under limited pilots.

## Reference

[15] H. Ye, G. Y. Li, and B.-H. Juang, “Power of deep learning for channel estimation and signal detection in OFDM systems,” *IEEE Wireless Communications Letters*, vol. 7, no. 1, pp. 114–117, Feb. 2018.

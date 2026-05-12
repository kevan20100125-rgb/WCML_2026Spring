from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DNN = ROOT / "DNN_Detection"
CHANNEL_ROOT = ROOT / "H_dataset"

sys.path.insert(0, str(DNN))

from ofdm_detection_core import (
    OfdmConfig,
    channel_covariance,
    detect_bits_from_channel,
    estimate_channel_lmmse,
    estimate_channel_ls,
    load_channel_dataset,
    make_pilot_config,
    simulate_receiver_blocks,
)


def main():
    txt_files = sorted(CHANNEL_ROOT.glob("[0-9]*.txt"), key=lambda p: int(p.stem))
    print(f"channel_txt_count={len(txt_files)}")
    if len(txt_files) != 400:
        raise SystemExit("Expected exactly 400 channel txt files.")

    cfg = OfdmConfig(k=64, cp=16, with_cp=True, clipping=False)
    train_channels, test_channels = load_channel_dataset(CHANNEL_ROOT, rows_per_file=8)
    print(f"train_channels={train_channels.shape}")
    print(f"test_channels={test_channels.shape}")

    cov = channel_covariance(train_channels, cfg.k)
    hermitian_error = float(np.max(np.abs(cov - cov.conj().T)))
    print(f"covariance_hermitian_error={hermitian_error:.3e}")
    if hermitian_error > 1e-8:
        raise SystemExit("Channel covariance is not Hermitian enough.")

    for pilots in [64, 8]:
        for method in ["ls", "lmmse"]:
            rng = np.random.default_rng(2026)
            pilot_cfg = make_pilot_config(cfg.k, pilots, "qpsk", seed=2026)
            errors = 0
            total = 0
            for _ in range(64):
                bits = rng.integers(0, 2, size=128, dtype=np.int64)
                channel_response = test_channels[rng.integers(0, test_channels.shape[0])]
                rx_pilot_freq, rx_data_freq = simulate_receiver_blocks(
                    bits, channel_response, 20, cfg, pilot_cfg, "qpsk", rng
                )
                if method == "ls":
                    h_est = estimate_channel_ls(rx_pilot_freq, pilot_cfg, cfg)
                else:
                    h_est = estimate_channel_lmmse(rx_pilot_freq, pilot_cfg, cfg, cov, 20)
                pred = detect_bits_from_channel(rx_data_freq, h_est, "qpsk")
                errors += int(np.sum(pred != bits))
                total += bits.size
            ber = errors / max(1, total)
            print(f"pilots={pilots} method={method} ber={ber:.6f}")

    print("validation_passed=True")


if __name__ == "__main__":
    main()

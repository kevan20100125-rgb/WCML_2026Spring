from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

PILOT_CONFIG_SEED = 314159
FEATURE_STD_EPS = 1e-6

from ofdm_detection_core import (
    OfdmConfig,
    bit_error_rate,
    channel_covariance,
    detect_bits_from_channel,
    estimate_channel_lmmse,
    estimate_channel_ls,
    load_channel_dataset,
    make_dnn_dataset,
    make_pilot_config,
    output_slices,
    simulate_receiver_blocks,
    total_bits_per_ofdm_symbol,
)


def require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required for the FC-DNN runner. Install it in your conda environment, "
            "then rerun this script."
        ) from exc
    return torch, nn, optim


def build_model(input_dim: int, output_dim: int, hidden_dims: Sequence[int]):
    torch, nn, _ = require_torch()
    layers: List[nn.Module] = []
    last = input_dim
    for hidden in hidden_dims:
        layers.append(nn.Linear(last, int(hidden)))
        layers.append(nn.ReLU())
        last = int(hidden)
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


def device_from_arg(device_arg: str):
    torch, _, _ = require_torch()
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def slice_name(target_slice: slice) -> str:
    return f"{target_slice.start}_{target_slice.stop}"


def model_path(output_dir: Path, modulation: str, mode: str, pilots: int, snr_db: float, group_id: int, target_slice: slice) -> Path:
    snr_text = str(snr_db).replace(".", "p")
    model_dir = ensure_dir(output_dir / "models" / modulation / mode / f"p{pilots}" / f"snr{snr_text}")
    return model_dir / f"group{group_id}_{slice_name(target_slice)}.pt"



def normalize_feature_matrix(features: np.ndarray, feature_mean: np.ndarray, feature_std: np.ndarray) -> np.ndarray:
    mean = np.asarray(feature_mean, dtype=np.float32).reshape(1, -1)
    std = np.asarray(feature_std, dtype=np.float32).reshape(1, -1)
    std = np.where(std < FEATURE_STD_EPS, 1.0, std).astype(np.float32)
    return ((features.astype(np.float32) - mean) / std).astype(np.float32)


def feature_stats(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0).astype(np.float32)
    std = features.std(axis=0).astype(np.float32)
    std = np.where(std < FEATURE_STD_EPS, 1.0, std).astype(np.float32)
    return mean, std


def apply_checkpoint_feature_normalization(features: np.ndarray, payload: Dict[str, object]) -> np.ndarray:
    if payload.get("feature_normalized") and "feature_mean" in payload and "feature_std" in payload:
        return normalize_feature_matrix(
            features,
            np.asarray(payload["feature_mean"], dtype=np.float32),
            np.asarray(payload["feature_std"], dtype=np.float32),
        )
    print(
        "warning: checkpoint has no feature normalization statistics; "
        "evaluation is using raw features. Retrain with --force."
    )
    return features.astype(np.float32)


def train_one_dnn(
    train_channels: np.ndarray,
    cfg: OfdmConfig,
    modulation: str,
    pilots: int,
    snr_db: float,
    target_slice: slice,
    hidden_dims: Sequence[int],
    train_samples: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    device,
    checkpoint_path: Path,
    force: bool,
) -> Path:
    torch, nn, optim = require_torch()
    if checkpoint_path.exists() and not force:
        return checkpoint_path
    rng = np.random.default_rng(seed)
    pilot_cfg = make_pilot_config(cfg.k, pilots, modulation, seed=PILOT_CONFIG_SEED)
    x_train, y_train, _ = make_dnn_dataset(
        train_samples, train_channels, snr_db, cfg, pilot_cfg, modulation, target_slice, rng
    )
    feature_mean, feature_std = feature_stats(x_train)
    x_train = normalize_feature_matrix(x_train, feature_mean, feature_std)
    x_tensor = torch.from_numpy(x_train).to(device)
    y_tensor = torch.from_numpy(y_train).to(device)
    model = build_model(x_tensor.shape[1], y_tensor.shape[1], hidden_dims).to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    n = x_tensor.shape[0]
    for epoch in range(epochs):
        permutation = torch.randperm(n, device=device)
        total_loss = 0.0
        steps = 0
        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            logits = model(x_tensor[idx])
            loss = loss_fn(logits, y_tensor[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            steps += 1
        print(
            f"train modulation={modulation} mode_slice={target_slice.start}:{target_slice.stop} "
            f"pilots={pilots} snr={snr_db:g} epoch={epoch + 1}/{epochs} loss={total_loss / max(1, steps):.6f}"
        )
    ensure_dir(checkpoint_path.parent)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(x_tensor.shape[1]),
            "output_dim": int(y_tensor.shape[1]),
            "hidden_dims": list(map(int, hidden_dims)),
            "target_slice": [int(target_slice.start), int(target_slice.stop)],
            "modulation": modulation,
            "pilots": int(pilots),
            "snr_db": float(snr_db),
            "pilot_config_seed": int(PILOT_CONFIG_SEED),
            "feature_normalized": True,
            "feature_mean": feature_mean.astype(np.float32).tolist(),
            "feature_std": feature_std.astype(np.float32).tolist(),
        },
        checkpoint_path,
    )
    return checkpoint_path

def load_model(checkpoint_path: Path, device):
    torch, _, _ = require_torch()
    payload = torch.load(checkpoint_path, map_location=device)
    model = build_model(payload["input_dim"], payload["output_dim"], payload["hidden_dims"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def evaluate_dnn_condition(
    test_channels: np.ndarray,
    cfg: OfdmConfig,
    modulation: str,
    pilots: int,
    snr_db: float,
    mode: str,
    checkpoints: Sequence[Path],
    test_samples: int,
    seed: int,
    device,
) -> float:
    torch, _, _ = require_torch()
    rng = np.random.default_rng(seed)
    pilot_cfg = make_pilot_config(cfg.k, pilots, modulation, seed=PILOT_CONFIG_SEED)
    total_bits = total_bits_per_ofdm_symbol(cfg.k, modulation)
    features = np.empty((test_samples, 4 * cfg.k), dtype=np.float32)
    reference_bits = np.empty((test_samples, total_bits), dtype=np.int64)
    for idx in range(test_samples):
        bits = rng.integers(0, 2, size=total_bits, dtype=np.int64)
        channel_response = test_channels[rng.integers(0, test_channels.shape[0])]
        from ofdm_detection_core import simulate_dnn_features
        features[idx] = simulate_dnn_features(bits, channel_response, snr_db, cfg, pilot_cfg, modulation, rng)
        reference_bits[idx] = bits
    predicted_bits = np.zeros_like(reference_bits)
    with torch.no_grad():
        for group_id, checkpoint in enumerate(checkpoints):
            model, payload = load_model(checkpoint, device)
            start, stop = payload["target_slice"]
            x_eval = apply_checkpoint_feature_normalization(features, payload)
            x_tensor = torch.from_numpy(x_eval).to(device)
            logits = model(x_tensor)
            pred = (torch.sigmoid(logits) >= 0.5).detach().cpu().numpy().astype(np.int64)
            predicted_bits[:, start:stop] = pred
            group_ber = bit_error_rate(reference_bits[:, start:stop], pred)
            print(json.dumps({
                "diagnostic": "per_group_eval",
                "group_id": group_id,
                "slice_start": int(start),
                "slice_stop": int(stop),
                "modulation": modulation,
                "mode": mode,
                "pilots": int(pilots),
                "snr_db": float(snr_db),
                "group_ber": group_ber,
                "feature_normalized": bool(payload.get("feature_normalized", False)),
                "pilot_config_seed": payload.get("pilot_config_seed", None),
            }, sort_keys=True))
    return bit_error_rate(reference_bits, predicted_bits)

def evaluate_traditional_condition(
    test_channels: np.ndarray,
    train_channels: np.ndarray,
    cfg: OfdmConfig,
    modulation: str,
    pilots: int,
    snr_db: float,
    test_samples: int,
    seed: int,
    method: str,
    covariance_cache: Dict[str, np.ndarray],
) -> float:
    if pilots == 0:
        return math.nan
    rng = np.random.default_rng(seed)
    pilot_cfg = make_pilot_config(cfg.k, pilots, modulation, seed=PILOT_CONFIG_SEED)
    covariance = None
    if method == "lmmse":
        key = f"k{cfg.k}"
        if key not in covariance_cache:
            covariance_cache[key] = channel_covariance(train_channels, cfg.k)
        covariance = covariance_cache[key]
    errors = 0
    total = 0
    total_bits = total_bits_per_ofdm_symbol(cfg.k, modulation)
    for _ in range(test_samples):
        bits = rng.integers(0, 2, size=total_bits, dtype=np.int64)
        channel_response = test_channels[rng.integers(0, test_channels.shape[0])]
        rx_pilot_freq, rx_data_freq = simulate_receiver_blocks(
            bits, channel_response, snr_db, cfg, pilot_cfg, modulation, rng
        )
        if method == "ls":
            h_est = estimate_channel_ls(rx_pilot_freq, pilot_cfg, cfg)
        elif method == "lmmse":
            h_est = estimate_channel_lmmse(rx_pilot_freq, pilot_cfg, cfg, covariance, snr_db)
        else:
            raise ValueError(f"unknown traditional method: {method}")
        if h_est is None:
            return math.nan
        pred = detect_bits_from_channel(rx_data_freq, h_est, modulation)
        errors += int(np.sum(pred != bits))
        total += bits.size
    return errors / max(1, total)


def append_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "task",
        "method",
        "modulation",
        "mode",
        "pilots",
        "snr_db",
        "ber",
        "train_samples",
        "test_samples",
        "epochs",
        "notes",
    ]
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_plot(csv_path: Path, figure_path: Path, title: str) -> None:
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib or pandas is not available. CSV was written without a figure.")
        return
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        return
    ensure_dir(figure_path.parent)
    plt.figure()
    group_cols = ["task", "method", "modulation", "mode", "pilots"]
    for key, group in df.groupby(group_cols, dropna=False):
        label = ", ".join([f"{name}={value}" for name, value in zip(group_cols, key) if str(value) != "nan"])
        group = group.sort_values("snr_db")
        plt.semilogy(group["snr_db"], group["ber"], marker="o", label=label)
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title(title)
    plt.grid(True, which="both")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()


def run_dnn_grid(args, train_channels: np.ndarray, test_channels: np.ndarray, cfg: OfdmConfig, task: str, modulation: str, mode: str, pilots_list: Sequence[int], snrs: Sequence[float], hidden_dims: Sequence[int]) -> List[Dict[str, object]]:
    device = device_from_arg(args.device)
    rows: List[Dict[str, object]] = []
    for pilots in pilots_list:
        for snr_db in snrs:
            slices = output_slices(cfg.k, modulation, mode)
            if args.max_groups > 0:
                slices = slices[:args.max_groups]
            checkpoints: List[Path] = []
            for group_id, target_slice in enumerate(slices):
                ckpt = model_path(args.output_dir, modulation, mode, pilots, snr_db, group_id, target_slice)
                train_one_dnn(
                    train_channels=train_channels,
                    cfg=cfg,
                    modulation=modulation,
                    pilots=pilots,
                    snr_db=snr_db,
                    target_slice=target_slice,
                    hidden_dims=hidden_dims,
                    train_samples=args.train_samples,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    seed=args.seed + 10000 * group_id + 97 * pilots + int(10 * snr_db),
                    device=device,
                    checkpoint_path=ckpt,
                    force=args.force,
                )
                checkpoints.append(ckpt)
            if len(slices) != len(output_slices(cfg.k, modulation, mode)):
                note = "partial group smoke run, not final BER"
            else:
                note = "full DNN output coverage"
            ber = evaluate_dnn_condition(
                test_channels=test_channels,
                cfg=cfg,
                modulation=modulation,
                pilots=pilots,
                snr_db=snr_db,
                mode=mode,
                checkpoints=checkpoints,
                test_samples=args.test_samples,
                seed=args.seed + 777 + 89 * pilots + int(10 * snr_db),
                device=device,
            )
            row = {
                "task": task,
                "method": "fc_dnn",
                "modulation": modulation,
                "mode": mode,
                "pilots": pilots,
                "snr_db": snr_db,
                "ber": ber,
                "train_samples": args.train_samples,
                "test_samples": args.test_samples,
                "epochs": args.epochs,
                "notes": note,
            }
            print(json.dumps(row, sort_keys=True))
            rows.append(row)
    return rows


def run_baseline_grid(args, train_channels: np.ndarray, test_channels: np.ndarray, cfg: OfdmConfig, task: str, modulation: str, pilots_list: Sequence[int], snrs: Sequence[float]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    covariance_cache: Dict[str, np.ndarray] = {}
    if args.skip_baselines:
        return rows
    for method in ["ls", "lmmse"]:
        for pilots in pilots_list:
            for snr_db in snrs:
                ber = evaluate_traditional_condition(
                    test_channels=test_channels,
                    train_channels=train_channels,
                    cfg=cfg,
                    modulation=modulation,
                    pilots=pilots,
                    snr_db=snr_db,
                    test_samples=args.test_samples,
                    seed=args.seed + 3001 + 67 * pilots + int(10 * snr_db),
                    method=method,
                    covariance_cache=covariance_cache,
                )
                row = {
                    "task": task,
                    "method": method,
                    "modulation": modulation,
                    "mode": "traditional",
                    "pilots": pilots,
                    "snr_db": snr_db,
                    "ber": ber,
                    "train_samples": 0,
                    "test_samples": args.test_samples,
                    "epochs": 0,
                    "notes": "traditional baseline skipped for pilots=0" if math.isnan(ber) else "traditional baseline",
                }
                print(json.dumps(row, sort_keys=True))
                rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OFDM FC-DNN detection experiment runner")
    parser.add_argument("--task", choices=["sanity", "b", "c", "d", "all"], default="sanity")
    parser.add_argument("--channel-root", type=Path, default=Path("../H_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("../outputs/ofdm_detection"))
    parser.add_argument("--snrs", type=float, nargs="+", default=[5, 10, 15, 20, 25])
    parser.add_argument("--pilots", type=int, nargs="+", default=[64, 8])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--test-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--channel-rows-per-file", type=int, default=128)
    parser.add_argument("--hidden", type=int, nargs="+", default=[500, 250, 120])
    parser.add_argument("--single-hidden", type=int, nargs="+", default=[1000, 500, 250])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--overwrite-results", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--max-groups", type=int, default=0, help="Use 0 for all groups. Use 1 only for smoke tests.")
    parser.add_argument("--without-cp",action="store_true",help="Run the OFDM simulation with CP removed.")
    return parser.parse_args()


def task_plan(args) -> List[Tuple[str, str, str, Sequence[int], Sequence[float], Sequence[int]]]:
    if args.task == "sanity":
        return [("sanity", "qpsk", "eight", [64], [20], args.hidden)]
    plan: List[Tuple[str, str, str, Sequence[int], Sequence[float], Sequence[int]]] = []
    if args.task in {"b", "all"}:
        plan.append(("b", "qpsk", "eight", args.pilots, args.snrs, args.hidden))
    if args.task in {"c", "all"}:
        plan.append(("c", "qam64", "eight", args.pilots, args.snrs, args.hidden))
    if args.task in {"d", "all"}:
        plan.append(("d", "qpsk", "eight", args.pilots, args.snrs, args.hidden))
        plan.append(("d", "qpsk", "single", args.pilots, args.snrs, args.single_hidden))
    return plan


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    cfg = OfdmConfig(k=64, cp=16, with_cp=not args.without_cp, clipping=False)
    print(f"loading channels from {args.channel_root}")
    train_channels, test_channels = load_channel_dataset(
        args.channel_root,
        rows_per_file=args.channel_rows_per_file,
    )
    print(f"loaded train_channels={train_channels.shape} test_channels={test_channels.shape}")
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "ofdm": asdict(cfg),
                "train_channels": list(train_channels.shape),
                "test_channels": list(test_channels.shape),
            },
            indent=2,
            sort_keys=True,
        )
    )
    all_rows: List[Dict[str, object]] = []
    for task, modulation, mode, pilots, snrs, hidden_dims in task_plan(args):
        if mode == "eight":
            all_rows.extend(run_baseline_grid(args, train_channels, test_channels, cfg, task, modulation, pilots, snrs))
        all_rows.extend(run_dnn_grid(args, train_channels, test_channels, cfg, task, modulation, mode, pilots, snrs, hidden_dims))
    csv_path = args.output_dir / f"results_{args.task}.csv"
    if args.overwrite_results:
        stale_paths = [
            csv_path,
            args.output_dir / f"ber_{args.task}.png",
            args.output_dir / "figures" / f"ber_{args.task}.png",
            args.output_dir / "figures" / f"ber_{args.task}.pdf",
        ]
        for stale_path in stale_paths:
            if stale_path.exists():
                stale_path.unlink()
    append_csv(csv_path, all_rows)
    write_plot(csv_path, args.output_dir / f"ber_{args.task}.png", f"OFDM detection task {args.task} BER")
    print(f"wrote {csv_path}")
    print(f"wrote figure if plotting dependencies are available: {args.output_dir / f'ber_{args.task}.png'}")


if __name__ == "__main__":
    main()

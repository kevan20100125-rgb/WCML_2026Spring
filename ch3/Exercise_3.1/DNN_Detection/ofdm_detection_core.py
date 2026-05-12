from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class OfdmConfig:
    k: int = 64
    cp: int = 16
    with_cp: bool = True
    clipping: bool = False
    clipping_ratio: float = 1.0


@dataclass(frozen=True)
class PilotConfig:
    pilot_carriers: np.ndarray
    data_carriers: np.ndarray
    pilot_values: np.ndarray


QAM64_GRAY_LEVELS: Dict[Tuple[int, int, int], int] = {
    (0, 0, 0): -7,
    (0, 0, 1): -5,
    (0, 1, 1): -3,
    (0, 1, 0): -1,
    (1, 1, 0): 1,
    (1, 1, 1): 3,
    (1, 0, 1): 5,
    (1, 0, 0): 7,
}
QAM64_LEVEL_TO_GRAY: Dict[int, Tuple[int, int, int]] = {v: k for k, v in QAM64_GRAY_LEVELS.items()}
QAM64_LEVEL_ARRAY = np.asarray([-7, -5, -3, -1, 1, 3, 5, 7], dtype=np.float64)


def bits_per_symbol(modulation: str) -> int:
    modulation = normalize_modulation(modulation)
    if modulation == "qpsk":
        return 2
    if modulation == "qam64":
        return 6
    raise ValueError(f"unsupported modulation: {modulation}")


def normalize_modulation(modulation: str) -> str:
    name = modulation.strip().lower().replace("-", "").replace("_", "")
    if name in {"qpsk", "4qam", "qam4"}:
        return "qpsk"
    if name in {"64qam", "qam64"}:
        return "qam64"
    raise ValueError(f"unsupported modulation: {modulation}")


def total_bits_per_ofdm_symbol(k: int, modulation: str) -> int:
    return k * bits_per_symbol(modulation)


def output_slices(k: int, modulation: str, mode: str, num_parallel: int = 8) -> List[slice]:
    total_bits = total_bits_per_ofdm_symbol(k, modulation)
    if mode == "single":
        return [slice(0, total_bits)]
    if mode != "eight":
        raise ValueError(f"unsupported DNN mode: {mode}")
    if total_bits % num_parallel != 0:
        raise ValueError("total output bits must be divisible by the number of parallel DNNs")
    group_size = total_bits // num_parallel
    return [slice(i * group_size, (i + 1) * group_size) for i in range(num_parallel)]


def modulate_bits(bits: np.ndarray, modulation: str) -> np.ndarray:
    modulation = normalize_modulation(modulation)
    bits = np.asarray(bits, dtype=np.int64).reshape(-1)
    mu = bits_per_symbol(modulation)
    if bits.size % mu != 0:
        raise ValueError(f"bit length {bits.size} is not divisible by {mu}")
    grouped = bits.reshape(-1, mu)
    if modulation == "qpsk":
        symbols = (2 * grouped[:, 0] - 1) + 1j * (2 * grouped[:, 1] - 1)
        return symbols.astype(np.complex128) / np.sqrt(2.0)
    if modulation == "qam64":
        levels_i = np.asarray([QAM64_GRAY_LEVELS[tuple(row[:3])] for row in grouped], dtype=np.float64)
        levels_q = np.asarray([QAM64_GRAY_LEVELS[tuple(row[3:])] for row in grouped], dtype=np.float64)
        return (levels_i + 1j * levels_q).astype(np.complex128) / np.sqrt(42.0)
    raise ValueError(f"unsupported modulation: {modulation}")


def _nearest_qam64_level(values: np.ndarray) -> np.ndarray:
    scaled = np.asarray(values, dtype=np.float64) * np.sqrt(42.0)
    distances = np.abs(scaled.reshape(-1, 1) - QAM64_LEVEL_ARRAY.reshape(1, -1))
    return QAM64_LEVEL_ARRAY[np.argmin(distances, axis=1)].astype(np.int64)


def demodulate_symbols(symbols: np.ndarray, modulation: str) -> np.ndarray:
    modulation = normalize_modulation(modulation)
    symbols = np.asarray(symbols, dtype=np.complex128).reshape(-1)
    if modulation == "qpsk":
        out = np.zeros((symbols.size, 2), dtype=np.int64)
        out[:, 0] = (np.real(symbols) >= 0).astype(np.int64)
        out[:, 1] = (np.imag(symbols) >= 0).astype(np.int64)
        return out.reshape(-1)
    if modulation == "qam64":
        levels_i = _nearest_qam64_level(np.real(symbols))
        levels_q = _nearest_qam64_level(np.imag(symbols))
        out = np.zeros((symbols.size, 6), dtype=np.int64)
        for idx, (li, lq) in enumerate(zip(levels_i, levels_q)):
            out[idx, :3] = QAM64_LEVEL_TO_GRAY[int(li)]
            out[idx, 3:] = QAM64_LEVEL_TO_GRAY[int(lq)]
        return out.reshape(-1)
    raise ValueError(f"unsupported modulation: {modulation}")


def make_pilot_config(k: int, num_pilots: int, modulation: str, seed: int = 1234) -> PilotConfig:
    if num_pilots < 0 or num_pilots > k:
        raise ValueError("num_pilots must be in [0, k]")
    all_carriers = np.arange(k)
    if num_pilots == 0:
        return PilotConfig(np.asarray([], dtype=np.int64), all_carriers, np.asarray([], dtype=np.complex128))
    if num_pilots == k:
        pilot_carriers = all_carriers
    else:
        step = k // num_pilots
        if step <= 0 or k % num_pilots != 0:
            raise ValueError("num_pilots must evenly divide 64 for this exercise runner")
        pilot_carriers = all_carriers[::step]
    data_carriers = np.setdiff1d(all_carriers, pilot_carriers, assume_unique=True)
    rng = np.random.default_rng(seed + 1009 * num_pilots + 9176 * bits_per_symbol(modulation))
    pilot_bits = rng.integers(0, 2, size=pilot_carriers.size * bits_per_symbol(modulation), dtype=np.int64)
    pilot_values = modulate_bits(pilot_bits, modulation)
    return PilotConfig(pilot_carriers, data_carriers, pilot_values)


def load_channel_file(path: Path, max_rows: int = 0) -> np.ndarray:
    data = np.loadtxt(path, max_rows=max_rows if max_rows and max_rows > 0 else None)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] % 2 != 0:
        raise ValueError(f"channel file must contain real and imaginary halves: {path}")
    half = data.shape[1] // 2
    return data[:, :half].astype(np.float64) + 1j * data[:, half:].astype(np.float64)


def load_channel_dataset(root: str | Path, train_high: int = 300, test_low: int = 301, test_high: int = 400, rows_per_file: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"channel root not found: {root}")
    txt_files = sorted(root.glob("[0-9]*.txt"), key=lambda p: int(p.stem))
    if not txt_files:
        raise FileNotFoundError(
            "no channel txt files found. Extract H_dataset.zip.001 to H_dataset.zip.004 before running."
        )
    train_blocks: List[np.ndarray] = []
    test_blocks: List[np.ndarray] = []
    for idx in range(1, train_high + 1):
        path = root / f"{idx}.txt"
        if not path.exists():
            raise FileNotFoundError(f"missing training channel file: {path}")
        train_blocks.append(load_channel_file(path, max_rows=rows_per_file))
    for idx in range(test_low, test_high + 1):
        path = root / f"{idx}.txt"
        if not path.exists():
            raise FileNotFoundError(f"missing testing channel file: {path}")
        test_blocks.append(load_channel_file(path, max_rows=rows_per_file))
    train_channels = np.concatenate(train_blocks, axis=0)
    test_channels = np.concatenate(test_blocks, axis=0)
    return train_channels, test_channels


def clip_signal(x: np.ndarray, clipping_ratio: float) -> np.ndarray:
    rms = np.sqrt(np.mean(np.abs(x) ** 2))
    threshold = clipping_ratio * rms
    y = np.array(x, copy=True)
    mask = np.abs(y) > threshold
    y[mask] = y[mask] * threshold / np.abs(y[mask])
    return y


def add_cp(ofdm_time: np.ndarray, cfg: OfdmConfig, modulation: str, rng: np.random.Generator) -> np.ndarray:
    if cfg.with_cp:
        prefix = ofdm_time[-cfg.cp:]
    else:
        noise_bits = rng.integers(0, 2, size=cfg.k * bits_per_symbol(modulation), dtype=np.int64)
        prefix_source = np.fft.ifft(modulate_bits(noise_bits, modulation))
        prefix = prefix_source[-cfg.cp:]
    return np.concatenate([prefix, ofdm_time])


def transmit_frequency_block(freq_block: np.ndarray, channel_response: np.ndarray, snr_db: float, cfg: OfdmConfig, modulation: str, rng: np.random.Generator) -> np.ndarray:
    time_block = np.fft.ifft(freq_block)
    tx = add_cp(time_block, cfg, modulation, rng)
    if cfg.clipping:
        tx = clip_signal(tx, cfg.clipping_ratio)
    convolved = np.convolve(tx, channel_response)
    signal_power = np.mean(np.abs(convolved) ** 2)
    noise_var = signal_power * 10.0 ** (-snr_db / 10.0)
    noise = np.sqrt(noise_var / 2.0) * (rng.standard_normal(convolved.shape) + 1j * rng.standard_normal(convolved.shape))
    rx = convolved + noise
    rx_no_cp = rx[cfg.cp: cfg.cp + cfg.k]
    return rx_no_cp


def build_pilot_frequency_block(pilot_cfg: PilotConfig, cfg: OfdmConfig, modulation: str, rng: np.random.Generator) -> np.ndarray:
    freq = np.zeros(cfg.k, dtype=np.complex128)
    if pilot_cfg.pilot_carriers.size:
        freq[pilot_cfg.pilot_carriers] = pilot_cfg.pilot_values
    if pilot_cfg.data_carriers.size:
        random_bits = rng.integers(0, 2, size=pilot_cfg.data_carriers.size * bits_per_symbol(modulation), dtype=np.int64)
        freq[pilot_cfg.data_carriers] = modulate_bits(random_bits, modulation)
    return freq


def build_data_frequency_block(bits: np.ndarray, cfg: OfdmConfig, modulation: str) -> np.ndarray:
    symbols = modulate_bits(bits, modulation)
    if symbols.size != cfg.k:
        raise ValueError(f"expected {cfg.k} symbols, got {symbols.size}")
    return symbols.astype(np.complex128)


def simulate_dnn_features(bits: np.ndarray, channel_response: np.ndarray, snr_db: float, cfg: OfdmConfig, pilot_cfg: PilotConfig, modulation: str, rng: np.random.Generator) -> np.ndarray:
    pilot_freq = build_pilot_frequency_block(pilot_cfg, cfg, modulation, rng)
    data_freq = build_data_frequency_block(bits, cfg, modulation)
    rx_pilot = transmit_frequency_block(pilot_freq, channel_response, snr_db, cfg, modulation, rng)
    rx_data = transmit_frequency_block(data_freq, channel_response, snr_db, cfg, modulation, rng)
    return np.concatenate([np.real(rx_pilot), np.imag(rx_pilot), np.real(rx_data), np.imag(rx_data)]).astype(np.float32)


def simulate_receiver_blocks(bits: np.ndarray, channel_response: np.ndarray, snr_db: float, cfg: OfdmConfig, pilot_cfg: PilotConfig, modulation: str, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    pilot_freq = build_pilot_frequency_block(pilot_cfg, cfg, modulation, rng)
    data_freq = build_data_frequency_block(bits, cfg, modulation)
    rx_pilot_time = transmit_frequency_block(pilot_freq, channel_response, snr_db, cfg, modulation, rng)
    rx_data_time = transmit_frequency_block(data_freq, channel_response, snr_db, cfg, modulation, rng)
    return np.fft.fft(rx_pilot_time), np.fft.fft(rx_data_time)


def frequency_response(channel_response: np.ndarray, k: int) -> np.ndarray:
    return np.fft.fft(channel_response, n=k)


def estimate_channel_ls(rx_pilot_freq: np.ndarray, pilot_cfg: PilotConfig, cfg: OfdmConfig) -> np.ndarray | None:
    if pilot_cfg.pilot_carriers.size == 0:
        return None
    h_pilots = rx_pilot_freq[pilot_cfg.pilot_carriers] / pilot_cfg.pilot_values
    if pilot_cfg.pilot_carriers.size == cfg.k:
        return h_pilots
    x = pilot_cfg.pilot_carriers.astype(np.float64)
    xi = np.arange(cfg.k, dtype=np.float64)
    h_real = np.interp(xi, x, np.real(h_pilots))
    h_imag = np.interp(xi, x, np.imag(h_pilots))
    return h_real + 1j * h_imag


def channel_covariance(channels: np.ndarray, k: int) -> np.ndarray:
    responses = np.fft.fft(channels, n=k, axis=1)
    responses = responses - responses.mean(axis=0, keepdims=True)
    return (responses.T @ responses.conj()) / max(1, responses.shape[0] - 1)


def estimate_channel_lmmse(rx_pilot_freq: np.ndarray, pilot_cfg: PilotConfig, cfg: OfdmConfig, covariance: np.ndarray, snr_db: float) -> np.ndarray | None:
    if pilot_cfg.pilot_carriers.size == 0:
        return None
    h_ls_p = rx_pilot_freq[pilot_cfg.pilot_carriers] / pilot_cfg.pilot_values
    idx = pilot_cfg.pilot_carriers
    r_hp = covariance[:, idx]
    r_pp = covariance[np.ix_(idx, idx)]
    noise_var = 10.0 ** (-snr_db / 10.0)
    diagonal_scale = float(np.real(np.trace(r_pp)) / max(1, idx.size))
    jitter = max(1e-10, 1e-8 * diagonal_scale)
    regularizer = (noise_var + jitter) * np.eye(idx.size, dtype=np.complex128)
    return r_hp @ np.linalg.solve(r_pp + regularizer, h_ls_p)


def detect_bits_from_channel(rx_data_freq: np.ndarray, h_est: np.ndarray, modulation: str) -> np.ndarray:
    safe_h = np.where(np.abs(h_est) < 1e-9, 1e-9 + 0j, h_est)
    equalized = rx_data_freq / safe_h
    return demodulate_symbols(equalized, modulation)


def bit_error_rate(reference_bits: np.ndarray, predicted_bits: np.ndarray) -> float:
    reference_bits = np.asarray(reference_bits, dtype=np.int64).reshape(-1)
    predicted_bits = np.asarray(predicted_bits, dtype=np.int64).reshape(-1)
    if reference_bits.shape != predicted_bits.shape:
        raise ValueError(f"BER shape mismatch: {reference_bits.shape} vs {predicted_bits.shape}")
    return float(np.mean(reference_bits != predicted_bits))


def make_dnn_dataset(n_samples: int, channels: np.ndarray, snr_db: float, cfg: OfdmConfig, pilot_cfg: PilotConfig, modulation: str, target_slice: slice, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_bits = total_bits_per_ofdm_symbol(cfg.k, modulation)
    x = np.empty((n_samples, 4 * cfg.k), dtype=np.float32)
    y = np.empty((n_samples, target_slice.stop - target_slice.start), dtype=np.float32)
    full_bits = np.empty((n_samples, total_bits), dtype=np.int64)
    for idx in range(n_samples):
        bits = rng.integers(0, 2, size=total_bits, dtype=np.int64)
        channel_response = channels[rng.integers(0, channels.shape[0])]
        x[idx] = simulate_dnn_features(bits, channel_response, snr_db, cfg, pilot_cfg, modulation, rng)
        y[idx] = bits[target_slice].astype(np.float32)
        full_bits[idx] = bits
    return x, y, full_bits

import os
import numpy as np


def generate_channel_dataset(num_samples, channel_len=16, decay_factor=3.0, seed=1):
    rng = np.random.default_rng(seed)

    # exponential power delay profile
    tap_idx = np.arange(channel_len, dtype=np.float64)
    pdp = np.exp(-tap_idx / decay_factor)
    pdp = pdp / np.sum(pdp)

    real = rng.standard_normal((num_samples, channel_len))
    imag = rng.standard_normal((num_samples, channel_len))
    h = (real + 1j * imag) * np.sqrt(pdp.reshape(1, -1) / 2.0)

    # normalize each channel realization to unit average power
    power = np.sum(np.abs(h) ** 2, axis=1, keepdims=True)
    h = h / np.sqrt(np.maximum(power, 1e-12))

    return h.astype(np.complex64)


if __name__ == "__main__":
    train_size = 100000
    test_size = 390000
    channel_len = 16

    os.makedirs("tools", exist_ok=True)

    channel_train = generate_channel_dataset(
        num_samples=train_size,
        channel_len=channel_len,
        decay_factor=3.0,
        seed=1,
    )
    channel_test = generate_channel_dataset(
        num_samples=test_size,
        channel_len=channel_len,
        decay_factor=3.0,
        seed=2,
    )

    np.save(os.path.join("tools", "channel_train.npy"), channel_train)
    np.save(os.path.join("tools", "channel_test.npy"), channel_test)

    print("Saved:")
    print(os.path.join("tools", "channel_train.npy"), channel_train.shape, channel_train.dtype)
    print(os.path.join("tools", "channel_test.npy"), channel_test.shape, channel_test.dtype)
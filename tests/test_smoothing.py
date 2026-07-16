"""One-Euro filter behavior."""

import numpy as np

from pokertell.behavior.smoothing import OneEuroFilter, smooth_trajectory


def test_filter_reduces_noise_variance():
    # The chip-push scenario: steady wrist translation plus detector jitter.
    # At default settings the filter cuts error roughly 4x on this signal.
    rng = np.random.default_rng(0)
    t = np.linspace(0, 4, 120)
    clean = 0.5 * t
    noisy = clean + rng.normal(0, 0.2, len(t))

    smoothed = np.array(smooth_trajectory(list(noisy), freq=30.0))

    noise_before = np.mean((noisy - clean) ** 2)
    noise_after = np.mean((smoothed - clean) ** 2)
    assert noise_after < 0.5 * noise_before


def test_first_sample_passes_through():
    f = OneEuroFilter(freq=30.0)
    assert f(5.0) == 5.0


def test_constant_signal_is_unchanged():
    f = OneEuroFilter(freq=30.0)
    out = [f(1.0) for _ in range(20)]
    assert all(abs(v - 1.0) < 1e-9 for v in out)

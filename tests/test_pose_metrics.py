"""Motion smoothness metrics: smooth trajectories must score smoother."""

import numpy as np

from pokertell.behavior.pose import jerk_rms, spectral_arc_length, velocity_profile

FPS = 30.0


def _smooth_push(n=60):
    """Minimum-jerk-like straight push: sigmoid position profile."""
    t = np.linspace(-4, 4, n)
    x = 1 / (1 + np.exp(-t))
    y = np.zeros(n)
    return np.column_stack([x * 300, y])


def _jerky_push(n=60, seed=0):
    rng = np.random.default_rng(seed)
    traj = _smooth_push(n)
    traj[:, 0] += rng.normal(0, 5, n)
    traj[:, 1] += rng.normal(0, 5, n)
    return traj


def test_jerk_rms_orders_smooth_below_jerky():
    assert jerk_rms(_smooth_push(), FPS) < jerk_rms(_jerky_push(), FPS)


def test_sparc_orders_smooth_above_jerky():
    smooth = spectral_arc_length(velocity_profile(_smooth_push(), FPS), FPS)
    jerky = spectral_arc_length(velocity_profile(_jerky_push(), FPS), FPS)
    assert smooth > jerky  # both negative; closer to zero is smoother


def test_short_trajectory_returns_nan():
    assert np.isnan(jerk_rms(_smooth_push(3), FPS))

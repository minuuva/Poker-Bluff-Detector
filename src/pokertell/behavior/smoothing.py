"""One-Euro filter for keypoint smoothing.

Raw pose and face keypoints from per-frame detectors are jittery. Any motion
feature computed on raw keypoints (velocity, jerk, smoothness) will mostly
measure detector noise. Every trajectory must pass through this filter before
downstream feature extraction.

Reference: Casiez, Roussel and Vogel, "1-euro filter: a simple speed-based
low-pass filter for noisy input in interactive systems", CHI 2012.
"""

import math


def _smoothing_factor(cutoff: float, dt: float) -> float:
    r = 2.0 * math.pi * cutoff * dt
    return r / (r + 1.0)


class OneEuroFilter:
    """Scalar One-Euro filter. Instantiate one per coordinate channel.

    Args:
        freq: nominal sampling frequency in Hz (video fps).
        min_cutoff: minimum cutoff frequency. Lower values remove more jitter
            at the cost of lag on slow movements.
        beta: speed coefficient. Higher values reduce lag on fast movements.
        d_cutoff: cutoff for the derivative estimate.
    """

    def __init__(
        self,
        freq: float,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        if freq <= 0:
            raise ValueError("freq must be positive")
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        self._t_prev: float | None = None

    def __call__(self, x: float, t: float | None = None) -> float:
        """Filter one sample. Pass t (seconds) for irregular sampling."""
        if self._x_prev is None:
            self._x_prev = x
            self._t_prev = t
            return x

        if t is not None and self._t_prev is not None and t > self._t_prev:
            dt = t - self._t_prev
        else:
            dt = 1.0 / self.freq
        self._t_prev = t

        dx = (x - self._x_prev) / dt
        a_d = _smoothing_factor(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _smoothing_factor(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat


def smooth_trajectory(values: list[float], freq: float, **kwargs) -> list[float]:
    """Convenience wrapper: filter a whole 1D trajectory sampled at freq Hz."""
    f = OneEuroFilter(freq=freq, **kwargs)
    return [f(v) for v in values]

"""
filters.py — One Euro Filter for temporal smoothing of joint trajectories.

Reference: Casiez et al. 2012 — "1€ Filter: A Simple Speed-based
Low-pass Filter for Noisy Input in Interactive Systems"

Preferred over Savitzky-Golay for boxing because it adapts to motion speed:
  - Heavy smoothing during slow guard position (low speed → low cutoff)
  - Minimal lag during fast punch delivery (high speed → high cutoff)
"""

from __future__ import annotations

import numpy as np

from src.constants import NUM_ACTIVE_JOINTS


class OneEuroFilter:
    """
    Adaptive low-pass filter for a single scalar signal.

    Parameters
    ----------
    freq       : sampling frequency in Hz (e.g. 30 for 30fps video)
    min_cutoff : minimum cutoff frequency — lower = smoother at rest
    beta       : speed coefficient — higher = less lag during fast motion
    d_cutoff   : derivative cutoff frequency (default 1.0 Hz)

    Usage
    -----
    >>> f = OneEuroFilter(freq=30, min_cutoff=1.0, beta=0.1)
    >>> smoothed = f(raw_value)
    """

    def __init__(
        self,
        freq:       float = 30.0,
        min_cutoff: float = 1.0,
        beta:       float = 0.1,
        d_cutoff:   float = 1.0,
    ) -> None:
        self.freq       = freq
        self.min_cutoff = min_cutoff
        self.beta       = beta
        self.d_cutoff   = d_cutoff
        self._x_prev:  float | None = None
        self._dx_prev: float        = 0.0

    def _alpha(self, cutoff: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        te  = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float) -> float:
        if self._x_prev is None:
            self._x_prev = x
            return x

        dx     = (x - self._x_prev) * self.freq
        a_d    = self._alpha(self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a      = self._alpha(cutoff)
        x_hat  = a * x + (1.0 - a) * self._x_prev

        self._x_prev  = x_hat
        self._dx_prev = dx_hat
        return x_hat

    def reset(self) -> None:
        self._x_prev  = None
        self._dx_prev = 0.0


def make_joint_filters(
    freq:       float = 30.0,
    min_cutoff: float = 1.0,
    beta:       float = 0.1,
) -> dict[int, list[OneEuroFilter]]:
    """
    Create one OneEuroFilter per active joint per spatial dimension (x, y, z).

    Returns
    -------
    filters : dict mapping local joint index → [filter_x, filter_y, filter_z]

    Usage
    -----
    filters = make_joint_filters(freq=30.0)
    for t in range(T):
        for j in range(NUM_ACTIVE_JOINTS):
            for dim in range(3):
                seq[t, j, dim] = filters[j][dim](seq[t, j, dim])
    """
    return {
        j: [OneEuroFilter(freq=freq, min_cutoff=min_cutoff, beta=beta)
            for _ in range(3)]
        for j in range(NUM_ACTIVE_JOINTS)
    }
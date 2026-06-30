import numpy as np

from src.dataset import SR


def _freq_features(X: np.ndarray) -> np.ndarray:
    """Channel-wise frequency-domain features via rFFT along the time axis --
    the natural complement to the time-domain stats (captures rhythm/
    periodicity that min/max/std/percentiles can't see):
      spectral centroid   -- "center of mass" frequency (energy-weighted)
      dominant frequency  -- frequency bin with peak power
      spectral entropy    -- how peaked (low) vs. flat/noisy (high) the
                             spectrum is, normalized to [0, 1]
      band power (lo/mid/hi) -- energy fraction in three equal frequency
                             bands, summing to 1 (drop one to avoid collinearity)
    DC bin dropped throughout (per-infant z-scoring already centers windows,
    so it carries ~no signal and would dominate the energy-normalized terms).
    """
    T = X.shape[1]
    freqs = np.fft.rfftfreq(T, d=1.0 / SR)[1:]              # (F,) -- DC dropped
    psd   = np.abs(np.fft.rfft(X, axis=1))[:, 1:] ** 2      # (N, F, C)
    psd_n = psd / (psd.sum(axis=1, keepdims=True) + 1e-12)

    centroid = (psd_n * freqs[None, :, None]).sum(axis=1)
    dominant = freqs[psd.argmax(axis=1)]
    entropy  = -(psd_n * np.log(psd_n + 1e-12)).sum(axis=1) / np.log(len(freqs))

    n = len(freqs) // 3
    band_lo  = psd_n[:, :n].sum(axis=1)
    band_mid = psd_n[:, n:2 * n].sum(axis=1)
    return np.concatenate([centroid, dominant, entropy, band_lo, band_mid], axis=1)


def feature_extraction(X: np.ndarray) -> np.ndarray:
    """X: (N, T, C) windowed signal -> (N, n_feat) per-window feature vector.

    Channel-wise time-domain statistics (range, central tendency & spread,
    slope, energy, percentiles + IQR -- the standard shallow-baseline set)
    plus frequency-domain features from `_freq_features` (rhythm/periodicity).
    (Coefficient of variation skipped: signal is per-infant z-scored, so
    local window means sit near zero and std/mean is numerically degenerate.)
    """
    d = [
        X.min(axis=1), X.max(axis=1), np.ptp(X, axis=1),
        X.mean(axis=1), X.std(axis=1),
        np.diff(X, axis=1).mean(axis=1),
        np.abs(np.diff(X, axis=1)).mean(axis=1),
        np.sqrt((X ** 2).mean(axis=1)), (X ** 2).mean(axis=1),
    ]
    p = np.percentile(X, [10, 25, 50, 75, 90], axis=1)      # (5, N, C)
    d += [p[i] for i in range(len(p))]
    d.append(p[3] - p[1])                                    # IQR = p75 - p25
    d.append(_freq_features(X))
    return np.concatenate(d, axis=1)

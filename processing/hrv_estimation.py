import numpy as np
from scipy.signal import find_peaks, welch, medfilt
from scipy.interpolate import interp1d


def refine_peaks_quadratic(signal, peaks):
    """
    Sub-frame quadratic peak refinement
    """
    refined = []

    for p in peaks:
        if p <= 0 or p >= len(signal) - 1:
            continue

        y1 = signal[p - 1]
        y2 = signal[p]
        y3 = signal[p + 1]

        denom = (y1 - 2 * y2 + y3)
        if abs(denom) < 1e-8:
            refined.append(p)
            continue

        delta = 0.5 * (y1 - y3) / denom
        refined.append(p + delta)

    return np.array(refined)


def estimate_hrv(pulse_signal, fps):

    if len(pulse_signal) < fps * 30:
        return None

    # Normalize
    signal = pulse_signal - np.mean(pulse_signal)
    signal = signal / (np.std(signal) + 1e-8)

    # --------------------------
    # Strong Peak Detection
    # --------------------------
    min_distance = int(fps * 0.4)
    peaks, _ = find_peaks(
        signal,
        distance=min_distance,
        prominence=0.6
    )

    if len(peaks) < 15:
        return None

    # --------------------------
    # Sub-frame Refinement
    # --------------------------
    refined_peaks = refine_peaks_quadratic(signal, peaks)

    if len(refined_peaks) < 15:
        return None

    # --------------------------
    # RR intervals (ms)
    # --------------------------
    rr_intervals = np.diff(refined_peaks) / fps * 1000

    # Physiological bounds
    rr_intervals = rr_intervals[
        (rr_intervals > 400) & (rr_intervals < 1500)
    ]

    if len(rr_intervals) < 15:
        return None

    # Median smoothing
    rr_intervals = medfilt(rr_intervals, kernel_size=3)

    # Z-score filtering
    mean_rr = np.mean(rr_intervals)
    std_rr = np.std(rr_intervals)

    if std_rr == 0:
        return None

    z_scores = np.abs((rr_intervals - mean_rr) / std_rr)
    rr_intervals = rr_intervals[z_scores < 2.5]

    if len(rr_intervals) < 15:
        return None

    # --------------------------
    # Time Domain HRV
    # --------------------------
    mean_rr = np.mean(rr_intervals)
    sdnn = np.std(rr_intervals)
    rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
    pnn50 = (
        np.sum(np.abs(np.diff(rr_intervals)) > 50)
        / len(rr_intervals)
        * 100
    )

    # --------------------------
    # Frequency Domain HRV
    # --------------------------
    rr_time = np.cumsum(rr_intervals) / 1000
    rr_time -= rr_time[0]

    if rr_time[-1] < 20:
        return None

    try:
        interp_func = interp1d(rr_time, rr_intervals, kind='cubic')
        uniform_time = np.arange(0, rr_time[-1], 0.25)
        uniform_rr = interp_func(uniform_time)
    except:
        return None

    freqs, psd = welch(
        uniform_rr,
        fs=4.0,
        nperseg=min(256, len(uniform_rr))
    )

    lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
    hf_mask = (freqs >= 0.15) & (freqs <= 0.4)

    lf_power = np.trapz(psd[lf_mask], freqs[lf_mask])
    hf_power = np.trapz(psd[hf_mask], freqs[hf_mask])

    lf_hf_ratio = lf_power / hf_power if hf_power > 1e-6 else None

    return {
        "mean_rr_ms": float(mean_rr),
        "sdnn_ms": float(sdnn),
        "rmssd_ms": float(rmssd),
        "pnn50_percent": float(pnn50),
        "lf_power": float(lf_power),
        "hf_power": float(hf_power),
        "lf_hf_ratio": float(lf_hf_ratio) if lf_hf_ratio else None
    }
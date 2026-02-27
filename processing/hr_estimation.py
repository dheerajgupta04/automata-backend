# import numpy as np
# from scipy.signal import detrend


# def pos_projection(rgb_signal):
#     rgb_signal = np.array(rgb_signal)

#     mean_rgb = np.mean(rgb_signal, axis=0)
#     rgb_norm = rgb_signal / mean_rgb - 1

#     projection_matrix = np.array([[0, 1, -1],
#                                    [-2, 1, 1]])

#     S = np.dot(rgb_norm, projection_matrix.T)

#     std_0 = np.std(S[:, 0])
#     std_1 = np.std(S[:, 1])

#     if std_1 == 0:
#         return None

#     alpha = std_0 / std_1
#     pulse = S[:, 0] + alpha * S[:, 1]

#     pulse = detrend(pulse)
#     pulse = pulse - np.mean(pulse)

#     return pulse


# def quadratic_interpolation(freqs, spectrum, peak_index):
#     """
#     Improves frequency estimate using parabolic interpolation
#     """
#     if peak_index <= 0 or peak_index >= len(spectrum) - 1:
#         return freqs[peak_index]

#     alpha = spectrum[peak_index - 1]
#     beta = spectrum[peak_index]
#     gamma = spectrum[peak_index + 1]

#     denominator = (alpha - 2 * beta + gamma)
#     if denominator == 0:
#         return freqs[peak_index]

#     delta = 0.5 * (alpha - gamma) / denominator

#     refined_freq = freqs[peak_index] + delta * (freqs[1] - freqs[0])

#     return refined_freq


# def estimate_hr(rgb_signal, fps):

#     rgb_signal = np.array(rgb_signal)

#     # 20 second window
#     window_size = int(fps * 20)
#     step_size = int(fps * 10)

#     hr_values = []

#     for start in range(0, len(rgb_signal) - window_size, step_size):

#         window = rgb_signal[start:start + window_size]

#         pulse = pos_projection(window)

#         if pulse is None:
#             continue

#         # Zero-padded FFT for higher resolution
#         fft_size = 4096
#         fft = np.fft.rfft(pulse, n=fft_size)
#         freqs = np.fft.rfftfreq(fft_size, 1 / fps)

#         # Tighter physiological band (54–138 BPM)
#         mask = (freqs >= 0.9) & (freqs <= 2.3)

#         fft_band = np.abs(fft[mask])
#         freqs_band = freqs[mask]

#         if len(fft_band) < 3:
#             continue

#         peak_idx = np.argmax(fft_band)

#         # SNR check
#         peak_power = fft_band[peak_idx]
#         mean_power = np.mean(fft_band)

#         if mean_power == 0:
#             continue

#         snr = peak_power / mean_power

#         if snr < 1.8:  # slightly relaxed threshold
#             continue

#         # Quadratic peak interpolation
#         refined_freq = quadratic_interpolation(freqs_band, fft_band, peak_idx)

#         bpm = refined_freq * 60
#         hr_values.append(bpm)

#     # -------------------------
#     # Fallback if windows fail
#     # -------------------------
#     if len(hr_values) == 0:
#         pulse = pos_projection(rgb_signal)
#         if pulse is None:
#             return 0

#         fft_size = 4096
#         fft = np.fft.rfft(pulse, n=fft_size)
#         freqs = np.fft.rfftfreq(fft_size, 1 / fps)

#         mask = (freqs >= 0.9) & (freqs <= 2.3)

#         fft_band = np.abs(fft[mask])
#         freqs_band = freqs[mask]

#         if len(fft_band) == 0:
#             return 0

#         peak_idx = np.argmax(fft_band)
#         refined_freq = quadratic_interpolation(freqs_band, fft_band, peak_idx)

#         return float(refined_freq * 60)

#     # Final HR = median of windows
#     return float(np.median(hr_values))
import numpy as np
from scipy.signal import detrend
from scipy.signal import find_peaks


def pos_projection(rgb_signal):
    rgb_signal = np.array(rgb_signal)

    mean_rgb = np.mean(rgb_signal, axis=0)
    rgb_norm = rgb_signal / mean_rgb - 1

    projection_matrix = np.array([[0, 1, -1],
                                   [-2, 1, 1]])

    S = np.dot(rgb_norm, projection_matrix.T)

    std_0 = np.std(S[:, 0])
    std_1 = np.std(S[:, 1])

    if std_1 == 0:
        return None

    alpha = std_0 / std_1
    pulse = S[:, 0] + alpha * S[:, 1]

    pulse = detrend(pulse)
    pulse = pulse - np.mean(pulse)

    return pulse


def quadratic_interpolation(freqs, spectrum, peak_index):
    if peak_index <= 0 or peak_index >= len(spectrum) - 1:
        return freqs[peak_index]

    alpha = spectrum[peak_index - 1]
    beta = spectrum[peak_index]
    gamma = spectrum[peak_index + 1]

    denominator = (alpha - 2 * beta + gamma)
    if denominator == 0:
        return freqs[peak_index]

    delta = 0.5 * (alpha - gamma) / denominator

    refined_freq = freqs[peak_index] + delta * (freqs[1] - freqs[0])

    return refined_freq


def harmonic_correction(freqs, spectrum, fps):
    """
    Detect fundamental vs harmonic peaks
    """

    peaks, _ = find_peaks(spectrum)

    if len(peaks) == 0:
        return None

    # Sort peaks by amplitude
    peak_amplitudes = spectrum[peaks]
    sorted_indices = np.argsort(peak_amplitudes)[::-1]

    top_peaks = peaks[sorted_indices[:3]]  # top 3 peaks

    peak_freqs = freqs[top_peaks]

    # Check harmonic relationships
    for i in range(len(peak_freqs)):
        for j in range(len(peak_freqs)):
            if i == j:
                continue

            f1 = peak_freqs[i]
            f2 = peak_freqs[j]

            # if one is approx double the other (within 10%)
            if abs(f1 - 2 * f2) < 0.1:
                return min(f1, f2)

    # If no harmonic relation found, return strongest
    return peak_freqs[0]


def estimate_hr(rgb_signal, fps):

    rgb_signal = np.array(rgb_signal)

    window_size = int(fps * 20)
    step_size = int(fps * 10)

    hr_values = []

    for start in range(0, len(rgb_signal) - window_size, step_size):

        window = rgb_signal[start:start + window_size]

        pulse = pos_projection(window)
        if pulse is None:
            continue

        fft_size = 4096
        fft = np.fft.rfft(pulse, n=fft_size)
        freqs = np.fft.rfftfreq(fft_size, 1 / fps)

        mask = (freqs >= 0.9) & (freqs <= 2.3)

        fft_band = np.abs(fft[mask])
        freqs_band = freqs[mask]

        if len(fft_band) < 5:
            continue

        # SNR check
        peak_power = np.max(fft_band)
        mean_power = np.mean(fft_band)

        if mean_power == 0:
            continue

        snr = peak_power / mean_power
        if snr < 1.6:
            continue

        # Harmonic-aware selection
        fundamental_freq = harmonic_correction(freqs_band, fft_band, fps)

        if fundamental_freq is None:
            continue

        bpm = fundamental_freq * 60
        hr_values.append(bpm)

    # Fallback
    if len(hr_values) == 0:
        pulse = pos_projection(rgb_signal)
        if pulse is None:
            return 0

        fft_size = 4096
        fft = np.fft.rfft(pulse, n=fft_size)
        freqs = np.fft.rfftfreq(fft_size, 1 / fps)

        mask = (freqs >= 0.9) & (freqs <= 2.3)

        fft_band = np.abs(fft[mask])
        freqs_band = freqs[mask]

        fundamental_freq = harmonic_correction(freqs_band, fft_band, fps)

        if fundamental_freq is None:
            return 0

        return float(fundamental_freq * 60)

    return float(np.median(hr_values))

def get_pos_signal(rgb_signal):

    rgb = np.array(rgb_signal)

    # Normalize
    rgb = rgb / (np.mean(rgb, axis=0) + 1e-6)

    # POS projection
    X = rgb[:, 0]
    Y = rgb[:, 1]
    Z = rgb[:, 2]

    S1 = Y - Z
    S2 = Y + Z - 2 * X

    alpha = np.std(S1) / (np.std(S2) + 1e-6)

    pos = S1 - alpha * S2

    pos = pos - np.mean(pos)

    return pos
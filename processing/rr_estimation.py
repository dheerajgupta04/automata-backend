import numpy as np
from scipy.signal import butter, filtfilt, hilbert, detrend


# -------------------------------------------------
# Generic bandpass filter
# -------------------------------------------------
def bandpass(signal, fps, low, high, order=3):

    nyquist = 0.5 * fps
    low /= nyquist
    high /= nyquist

    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


# -------------------------------------------------
# Respiratory Rate Estimation (POS Envelope Method)
# -------------------------------------------------
def estimate_rr(pos_pulse_signal, fps):

    pulse = np.array(pos_pulse_signal)

    if len(pulse) < int(fps * 20):
        return None

    # -----------------------------
    # 1️⃣ Clean heart band first
    # -----------------------------
    heart_band = bandpass(pulse, fps, 0.9, 2.3)

    # -----------------------------
    # 2️⃣ Extract amplitude envelope
    # -----------------------------
    analytic_signal = hilbert(heart_band)
    envelope = np.abs(analytic_signal)

    envelope = detrend(envelope)
    envelope = (envelope - np.mean(envelope)) / (np.std(envelope) + 1e-6)

    # -----------------------------
    # 3️⃣ Sliding window RR
    # -----------------------------
    window_size = int(fps * 20)
    step_size = int(window_size * 0.5)

    rr_candidates = []

    for start in range(0, len(envelope) - window_size, step_size):

        segment = envelope[start:start + window_size]

        resp_band = bandpass(segment, fps, 0.15, 0.4)

        fft_size = 4096
        fft = np.fft.rfft(resp_band, n=fft_size)
        freqs = np.fft.rfftfreq(fft_size, 1 / fps)

        mask = (freqs >= 0.15) & (freqs <= 0.4)

        fft_band = np.abs(fft[mask])
        freqs_band = freqs[mask]

        if len(fft_band) == 0:
            continue

        peak_idx = np.argmax(fft_band)
        peak_power = fft_band[peak_idx]
        median_power = np.median(fft_band)

        # ✅ Relaxed SNR threshold
        if peak_power < 2 * median_power:
            continue

        peak_freq = freqs_band[peak_idx]
        rr = peak_freq * 60

        rr_candidates.append(rr)

    # -----------------------------
    # 4️⃣ If windows failed → fallback global
    # -----------------------------
    if len(rr_candidates) == 0:

        resp_band = bandpass(envelope, fps, 0.15, 0.4)

        fft = np.fft.rfft(resp_band, n=4096)
        freqs = np.fft.rfftfreq(4096, 1 / fps)

        mask = (freqs >= 0.15) & (freqs <= 0.4)
        fft_band = np.abs(fft[mask])
        freqs_band = freqs[mask]

        if len(fft_band) == 0:
            return None

        peak_freq = freqs_band[np.argmax(fft_band)]
        return round(float(peak_freq * 60), 2)

    # -----------------------------
    # 5️⃣ Final RR = Median of windows
    # -----------------------------
    final_rr = np.median(rr_candidates)

    return round(float(final_rr), 2)
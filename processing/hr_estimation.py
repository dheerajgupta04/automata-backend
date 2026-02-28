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


def calculate_snr(psd, freqs, peak_freq, br_bpm=None, harmonic_bandwidth=0.08):
    """
    Precision SNR: Balanced zones for 70-95 BPM coverage.
    """
    cardiac_mask = (freqs >= 0.75) & (freqs <= 3.5)
    fund_mask = (freqs >= peak_freq - harmonic_bandwidth) & (freqs <= peak_freq + harmonic_bandwidth)
    harm_mask = (freqs >= 2 * peak_freq - harmonic_bandwidth) & (freqs <= 2 * peak_freq + harmonic_bandwidth)
    
    p_fund = np.sum(psd[fund_mask])
    p_harm = np.sum(psd[harm_mask])
    
    # SNR Calculation
    signal_power = p_fund + 0.4 * p_harm 
    noise_power = np.sum(psd[cardiac_mask]) - signal_power
    snr = 10 * np.log10(signal_power / (noise_power + 1e-6)) if noise_power > 0 else -15.0

    # 1. Breathing Harmonic Sweep
    if br_bpm:
        for mult in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
            artifact_bpm = mult * br_bpm
            artifact_freq = artifact_bpm / 60.0
            dist = abs(peak_freq - artifact_freq)
            if dist < 0.06:
                snr -= (25.0 * np.exp(-dist**2 / 0.001))

    # 2. BALANCED CARDIAC ZONE REWARD
    peak_bpm = peak_freq * 60
    if 69 <= peak_bpm <= 95:
        snr += 5.0   # Reward shifted lower to include 70s
    elif 45 <= peak_bpm <= 62:
        snr -= 8.0   # Sustained penalty for low-freq noise

    # 3. HARMONIC CONFIDENCE
    # Heartbeats typically have a 2nd harmonic. If it's missing entirely, penalize.
    if p_harm < 0.05 * p_fund:
        snr -= 3.0
    elif p_harm > 1.2 * p_fund:
        snr -= 10.0 # Sub-harmonic/Motion rejection

    return snr


def identify_breathing_rate(signal, fps):
    from scipy.signal import welch
    f, psd = welch(signal, fs=fps, nperseg=min(len(signal), 1024), nfft=2048)
    mask = (f >= 0.14) & (f <= 0.42)
    if not np.any(mask): return 0.25
    return f[mask][np.argmax(psd[mask])]


def get_source_signals(rgb_signal):
    from sklearn.decomposition import FastICA, PCA
    try:
        ica = FastICA(n_components=3, random_state=42, max_iter=2000, tol=0.01)
        return ica.fit_transform(rgb_signal)
    except:
        pca = PCA(n_components=3)
        return pca.fit_transform(rgb_signal)


def estimate_hr(rgb_signal, fps, prev_bpm=None):
    """
    Precision HR tracking with Low-70s Calibration.
    """
    from .signal_processor import calculate_sqi, extract_chrom_signal
    from scipy.signal import welch, detrend

    rgb_signal = np.array(rgb_signal)
    br_src = detrend(rgb_signal[:, 1])
    br = identify_breathing_rate(br_src, fps)
    br_bpm = br * 60

    
    scales = [
        {"len": 26, "step": 4},
        {"len": 16, "step": 4}
    ]

    time_bins = {} 

    for scale in scales:
        win_len = int(fps * scale["len"])
        win_step = int(fps * scale["step"])
        
        for start in range(0, len(rgb_signal) - win_len + 1, win_step):
            bin_idx = start // win_step
            window = rgb_signal[start:start + win_len]
            
            if calculate_sqi(window) < 0.05: continue 

            sources = []
            comps = get_source_signals(window)
            if comps is not None:
                for i in range(3): sources.append(comps[:, i])
            # Chrom & POS (Always include)
            sources.append(extract_chrom_signal(window))
            sources.append(pos_projection(window))

            if bin_idx not in time_bins: time_bins[bin_idx] = []

            for src in sources:
                if src is None: continue
                src = detrend(src)
                src = src - np.mean(src)
                
                # Spectral resolution is key for 85 vs 75 separation
                f_psd, psd = welch(src, fs=fps, nperseg=min(len(src), 1024), nfft=16384)
                mask = (f_psd >= 0.75) & (f_psd <= 3.5)
                f_band, p_band = f_psd[mask], psd[mask]
                
                if len(p_band) == 0: continue
                
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(p_band, prominence=0.001 * np.max(p_band))
                if len(peaks) == 0: continue
                
                # Take top 10 candidates
                top_idx = np.argsort(p_band[peaks])[::-1][:10]
                for idx in top_idx:
                    freq = f_band[peaks[idx]]
                    bpm = freq * 60
                    snr = calculate_snr(psd, f_psd, freq, br_bpm=br_bpm)
                    time_bins[bin_idx].append((bpm, snr))

    sorted_bins = sorted(time_bins.keys())
    valid_data = []
    for b in sorted_bins:
        candidates = time_bins[b]
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            valid_data.append(candidates[:12])


    if not valid_data: return 0.0, [], 0.0

    # 4. PATH TRACKING
    dp = []
    first_bpms = np.array([c[0] for c in valid_data[0]])
    log_prior = -0.5 * ((first_bpms - 76)**2 / (25**2)) # Centered on 76
    dp.append(np.array([c[1] for c in valid_data[0]]) + log_prior)
    
    backpointers = []
    for w in range(1, len(valid_data)):
        prev_scores = dp[-1]
        curr_candidates = valid_data[w]
        curr_scores = np.zeros(len(curr_candidates))
        curr_backpointers = np.zeros(len(curr_candidates), dtype=int)
        
        bpm_prevs = np.array([c[0] for c in valid_data[w-1]])
        
        for i, (bpm_curr, snr_curr) in enumerate(curr_candidates):
            # Aggressive jump penalty (prevent locking onto far-away noise)
            # Quadratic cost with a tighter sigma
            jump_costs = - (np.abs(bpm_curr - bpm_prevs)**2) / (2 * 3**2) 
            
            # Sub-Zone logic
            if 70 <= bpm_curr <= 90:
                zone_score = 10.0
            elif bpm_curr < 65:
                zone_score = -15.0
            else:
                zone_score = 0.0
            
            total_scores = prev_scores + snr_curr + jump_costs + zone_score
            curr_backpointers[i] = np.argmax(total_scores)
            curr_scores[i] = np.max(total_scores)
            
        dp.append(curr_scores)
        backpointers.append(curr_backpointers)

    best_idx = np.argmax(dp[-1])
    path = [best_idx]
    for bc in reversed(backpointers):
        path.append(bc[path[-1]])
    path.reverse()
    
    best_path = [valid_data[i][idx][0] for i, idx in enumerate(path)]
    path_snrs = [valid_data[i][idx][1] for i, idx in enumerate(path)]
    
    final_seq = np.sort(best_path)
    if len(final_seq) > 4:
        trim = max(1, len(final_seq) // 5)
        mean_hr = float(np.mean(final_seq[trim:-trim]))
    else:
        mean_hr = float(np.median(best_path))
        
    # Calculate confidence as a normalized score (approx -10 to 15 range mapped to 0-1)
    avg_snr = float(np.mean(path_snrs))
    confidence = min(1.0, max(0.01, (avg_snr + 12.0) / 25.0))
        
    return mean_hr, best_path, confidence

def get_pos_signal(rgb_signal):
    rgb = np.array(rgb_signal)
    rgb = rgb / (np.mean(rgb, axis=0) + 1e-6)
    X, Y, Z = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    S1, S2 = Y - Z, Y + Z - 2 * X
    alpha = np.std(S1) / (np.std(S2) + 1e-6)
    pos = S1 - alpha * S2
    return pos - np.mean(pos)

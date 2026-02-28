import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from processing.hr_estimation import estimate_hr
from processing.signal_processor import calculate_sqi, extract_chrom_signal

def generate_synthetic_ppg(fps, duration_sec, bpm):
    """Generates a synthetic PPG-like signal."""
    t = np.linspace(0, duration_sec, int(fps * duration_sec))
    freq = bpm / 60.0
    # Base signal + some harmonics + noise
    signal = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(4 * np.pi * freq * t)
    # Add noise
    signal += 0.1 * np.random.normal(size=len(t))
    # Convert to 3-channel RGB (G is strongest, R/B weaker)
    rgb = np.column_stack([
        0.5 * signal + 120, # R
        1.0 * signal + 125, # G
        0.3 * signal + 115  # B
    ])
    return rgb

def test_hr_accuracy(target_bpm, duration=30, fps=30, sub_harmonic_amp=0.0, breathing_harmonic_amp=0.0):
    print(f"\n[*] Testing HR: {target_bpm} BPM (Sub: {sub_harmonic_amp}, BreathHarm: {breathing_harmonic_amp})")
    
    t = np.linspace(0, duration, int(fps * duration))
    freq = target_bpm / 60.0
    rr_bpm = 18.0
    rr_freq = rr_bpm / 60.0
    
    # Fundamental pulse + Breathing fundamental (for detector)
    signal = np.sin(2 * np.pi * freq * t)
    # Add breathing fundamental so identify_breathing_rate works
    signal += 0.5 * np.sin(2 * np.pi * rr_freq * t)
    
    # Add strong breathing harmonic (interference)
    if breathing_harmonic_amp > 0:
        signal += breathing_harmonic_amp * np.sin(2 * np.pi * (3 * rr_freq) * t)
        
    # Add sub-harmonic
    if sub_harmonic_amp > 0:
        signal += sub_harmonic_amp * np.sin(2 * np.pi * (freq/2) * t)
        
    rgb = np.column_stack([
        0.5 * signal + 120,
        1.0 * signal + 125,
        0.3 * signal + 115
    ])
    
    est_bpm = estimate_hr(rgb, fps)
    print(f"[+] Estimated BPM: {est_bpm:.2f}")
    
    error = abs(est_bpm - target_bpm)
    print(f"[+] Error: {error:.2f} BPM")
    
    if error < 5.0:
        print("[SUCCESS] Accuracy within range.")
        return True
    return False

if __name__ == "__main__":
    # Case 1: Clean 75 BPM
    c1 = test_hr_accuracy(75)
    
    # Case 2: 85 BPM with breathing harmonic interference (mimics RR=18 trigger at 54 BPM)
    # The harmonic at 54 BPM is 3x stronger than the heart rate at 85 BPM
    c2 = test_hr_accuracy(85, breathing_harmonic_amp=3.0)
    
    if c1 and c2:
        print("\n[ALL VERIFICATION TESTS PASSED]")
    else:
        print("\n[VERIFICATION FAILED]")

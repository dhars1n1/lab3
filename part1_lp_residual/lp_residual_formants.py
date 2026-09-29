"""
Q1 (b)(c)(d) - LP analysis of a 25 ms voiced frame
--------------------------------------------------
1. Load the recorded utterance (16 kHz, 16-bit .wav)
2. Automatically (or manually) pick a voiced 25 ms frame
3. Compute LPC coefficients (Levinson-Durbin) and the LP residual
   (inverse / error filtering:  e[n] = x[n] - sum_k a_k x[n-k])
4. Plot: (a) the voiced frame, (b) the LP residual, (c) LP spectrum
5. Estimate formant frequencies from the roots of the LPC polynomial

Usage:
    python lp_residual_formants.py my_name_is_utterance.wav
    python lp_residual_formants.py my_name_is_utterance.wav --start 0.42
"""

import argparse
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import freqz


# ---------------------------------------------------------------- #
# 1. LPC via Levinson-Durbin recursion
# ---------------------------------------------------------------- #
def autocorr(x, max_lag):
    n = len(x)
    r = np.zeros(max_lag + 1)
    for k in range(max_lag + 1):
        r[k] = np.sum(x[: n - k] * x[k:n])
    return r


def levinson_durbin(r, order):
    """Returns LPC coefficients a[1..order] such that
    x[n] ~= sum_{k=1}^{order} a[k] * x[n-k]   (prediction)
    and the prediction-error (gain) variance."""
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = r[0]
    for i in range(1, order + 1):
        acc = r[i] + np.sum(a[1:i] * r[i - 1:0:-1])
        k = -acc / e
        a_new = a.copy()
        a_new[i] = k
        a_new[1:i] = a[1:i] + k * a[i - 1:0:-1]
        a = a_new
        e *= (1 - k ** 2)
        if e <= 0:
            break
    return a, e   # a[0]=1, a[1..order] are the (negated-convention) coeffs


# ---------------------------------------------------------------- #
# 2. Voiced-frame picker (simple energy + zero-crossing heuristic)
# ---------------------------------------------------------------- #
def frame_periodicity(seg, sr, fmin=70, fmax=400):
    """Normalised autocorrelation peak in the pitch range (0..1).
    High value (> ~0.5) => strongly periodic => voiced."""
    seg = seg - np.mean(seg)
    r = np.correlate(seg, seg, mode="full")[len(seg) - 1:]
    if r[0] <= 1e-12:
        return 0.0
    r = r / r[0]
    lo, hi = int(sr / fmax), int(sr / fmin)
    return float(np.max(r[lo:hi]))


def pick_voiced_frame(x, sr, frame_len, start_time=None):
    """Pick the loudest frame that is genuinely voiced (periodic in the pitch
    range). x should already be DC-removed / pre-emphasised."""
    if start_time is not None:
        start = int(start_time * sr)
        return x[start:start + frame_len], start / sr

    hop = frame_len // 4
    cands = []
    for start in range(0, len(x) - frame_len, hop):
        seg = x[start:start + frame_len]
        cands.append((start, np.sum(seg ** 2), frame_periodicity(seg, sr)))
    max_e = max(c[1] for c in cands)
    voiced = [c for c in cands if c[2] > 0.6 and c[1] > 0.2 * max_e]
    pool = voiced if voiced else cands
    best = max(pool, key=lambda c: c[1] * c[2])
    return x[best[0]:best[0] + frame_len], best[0] / sr


# ---------------------------------------------------------------- #
# 3. Formant estimation from LPC roots
# ---------------------------------------------------------------- #
def estimate_formants(a, sr, min_bw_hz=None):
    """a: LPC coefficients with a[0]=1 (prediction convention above)."""
    # polynomial A(z) = 1 + a1 z^-1 + ... + ap z^-p  -> roots of the
    # polynomial with coefficients [1, a1, ..., ap]
    roots = np.roots(a)
    roots = roots[np.imag(roots) >= 0]  # keep upper half-plane only

    freqs, bandwidths = [], []
    for r in roots:
        angle = np.arctan2(np.imag(r), np.real(r))
        freq = angle * (sr / (2 * np.pi))
        bw = -0.5 * (sr / np.pi) * np.log(np.abs(r))
        if 90 < freq < sr / 2 - 90:   # discard near-DC / near-Nyquist roots
            freqs.append(freq)
            bandwidths.append(bw)

    order = np.argsort(freqs)
    freqs = np.array(freqs)[order]
    bandwidths = np.array(bandwidths)[order]

    if min_bw_hz is not None:
        keep = bandwidths < min_bw_hz
        freqs, bandwidths = freqs[keep], bandwidths[keep]

    return freqs, bandwidths


# ---------------------------------------------------------------- #
# main
# ---------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav_path")
    ap.add_argument("--start", type=float, default=None,
                     help="start time (s) of the 25 ms voiced frame; "
                          "if omitted, one is picked automatically")
    ap.add_argument("--order", type=int, default=None,
                     help="LPC order; default 16 for 16 kHz speech")
    ap.add_argument("--out_prefix", default="lp_analysis")
    args = ap.parse_args()

    sr = 16000
    x, _ = librosa.load(args.wav_path, sr=sr, mono=True)
    x = x - np.mean(x)                       # remove DC offset
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])   # pre-emphasis (flattens spectral tilt)
    x = x / (np.max(np.abs(x)) + 1e-9)

    frame_len = int(0.025 * sr)   # 25 ms
    frame, t0 = pick_voiced_frame(x, sr, frame_len, args.start)
    frame = frame * np.hamming(len(frame))

    order = args.order or 16   # typical for 16 kHz speech (2 coeffs per formant + margin)
    r = autocorr(frame, order)
    a, gain = levinson_durbin(r, order)

    # LP residual via inverse (error) filtering: e[n] = x[n] + sum a_k x[n-k]
    residual = np.convolve(frame, a, mode="same")

    # LP spectrum: 1 / A(z)
    w, h = freqz([1.0], a, worN=1024, fs=sr)
    lp_spectrum_db = 20 * np.log10(np.abs(h) + 1e-9)

    formants, bandwidths = estimate_formants(a, sr, min_bw_hz=700)

    # ---------------- plots ----------------
    fig, axes = plt.subplots(3, 1, figsize=(8, 9))

    t_axis = np.arange(len(frame)) / sr * 1000
    axes[0].plot(t_axis, frame)
    axes[0].set_title(f"25 ms voiced frame (start={t0*1000:.1f} ms)")
    axes[0].set_xlabel("Time (ms)"); axes[0].set_ylabel("Amplitude")

    axes[1].plot(t_axis, residual, color="darkorange")
    axes[1].set_title(f"LP residual (order={order})")
    axes[1].set_xlabel("Time (ms)"); axes[1].set_ylabel("Amplitude")

    axes[2].plot(w, lp_spectrum_db, label="LP spectrum (1/A(z))")
    for i, f in enumerate(formants[:4]):
        axes[2].axvline(f, color="red", linestyle="--", alpha=0.7)
        axes[2].text(f, axes[2].get_ylim()[1] - 5, f"F{i+1}={f:.0f} Hz",
                     rotation=90, va="top", fontsize=8, color="red")
    axes[2].set_title("LP spectrum with estimated formants")
    axes[2].set_xlabel("Frequency (Hz)"); axes[2].set_ylabel("Magnitude (dB)")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}.png", dpi=150)
    print(f"Saved plot -> {args.out_prefix}.png")

    print("\n--- Results ---")
    print(f"Frame start: {t0*1000:.1f} ms, LPC order: {order}")
    print("Estimated formants (Hz):", np.round(formants[:4], 1))
    print("Corresponding bandwidths (Hz):", np.round(bandwidths[:4], 1))
    print(f"Frame periodicity (voicing strength, 0-1): {frame_periodicity(frame, sr):.2f}")
    print(f"Residual energy / frame energy ratio: "
          f"{np.sum(residual**2) / np.sum(frame**2):.4f}")


if __name__ == "__main__":
    main()

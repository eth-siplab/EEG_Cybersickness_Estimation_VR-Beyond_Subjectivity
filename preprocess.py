"""Raw JULIETE recordings -> cached .npz datasets consumed by loader.py.

Reconstructs the (previously missing) preprocessing front-end described in the
paper "Beyond Subjectivity: Continuous Cybersickness Detection Using EEG-based
Multitaper Spectrum Estimation" (Sec. Pre-processing + Algorithm 1).

Per recording we read three per-stream exports that share a common Unity
millisecond clock.  The dataset is released as Google Sheets; a bulk Drive
download converts each to .xlsx (one per stream), and .csv is accepted too:

    <PID>_<COND>_EEG.xlsx          Millis, Hardware, <24 DSI-24 channels>
    <PID>_<COND>_Transforms.xlsx   Millis, HeadPosition_{X,Y,Z}, HeadRotation_{Yaw,Pitch,Roll}
    <PID>_<COND>_SubjectiveCs.xlsx  Millis, Rating, ChangePerSec   (joystick sickness level)

and emit  <PID>_<COND>.npz  holding a single pickled dict ``dataset`` with the
keys loader.py expects.  Storage orientation is chosen so loader's in-place
``transpose(0, 2, 1)`` on ``eeg``/``psd`` yields channel-major tensors, while
``psd_raw`` is already channel-major (loader uses it as-is):

    eeg            (n, T, 24)        filtered EEG @ 100 Hz, T = 300   -> loader -> (n, 24, T)
    psd            (n, F, 24)        log10 multitaper PSD             -> loader -> (n, 24, F)
    psd_raw        (n, 24, F)        linear multitaper PSD            (TR-PSD computed in loader)
    spectral_coeff (n, 24, 2)        [slope, intercept] of 1/f fit, 30-45 Hz
    freqs          (F,)             frequency axis of the PSD
    tf             list[n] (Li, 6)   head pose: pos xyz + rot ypr
    pth            list[n] (Li, 7)   Millis + head pose (loader takes norms of cols 1:4 / 4:7)
    joy            (n, 1)            per-segment joystick rating

The temporal-relative differencing (TR-PSD) and the kinematic 16-tuple are left
to loader.py, which already implements them; this script only produces psd_raw
and the raw per-segment streams.
"""

import os
import glob
import argparse
import numpy as np
import scipy.signal as sig
from scipy.signal.windows import dpss

import pandas as pd


def _read_table(path):
    """Read a per-recording export as a float array, .xlsx or .csv."""
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path).to_numpy(dtype=np.float64)
    return pd.read_csv(path).to_numpy(dtype=np.float64)


# ------------------------------------------------------------------ constants
FS_RAW = 300            # DSI-24 nominal sampling rate (Hz)
FS = 100               # target rate after resampling
WINDOW_SEC = 3          # non-overlapping analysis window
WIN = FS * WINDOW_SEC   # 300 samples per window
WIN_MS = WINDOW_SEC * 1000
N_EEG_CH = 24
NFFT = WIN              # 151 one-sided bins -> freqs 0..50 Hz
NW = 3                  # multitaper time-bandwidth product
K_TAPERS = 5            # number of Slepian (DPSS) tapers
ONEF_BAND = (30.0, 45.0)  # 1/f spectral-slope fit band

# DSI-24 channel order, matching analysis/analysis.py CHANNEL_NAMES
CHANNELS = "P3,C3,F3,Fz,F4,C4,P4,Cz,Pz,A1,Fp1,Fp2,T3,T5,O1,O2,X3,X2,F7,F8,X1,A2,T6,T4".split(",")


# ------------------------------------------------------------------ filtering
def _filter_eeg(raw):
    """raw EEG (N, 24) @ 300 Hz -> filtered (M, 24) @ 100 Hz.

    Swiss 50 Hz mains shows up as a ~5e5x single-bin spike in the raw EEG, so:
      1. notch 50 Hz (Q=5) -- removes the mains line, keeps the band around it;
      2. 4th-order Butterworth band-pass 1-49 Hz -- high-pass 1 Hz for drift,
         low-pass ~50 Hz to KEEP the 30-45 Hz 1/f band the paper uses as the
         cybersickness correlate (a 40 Hz cut would discard the top of it);
      3. resample 300 -> 100 Hz with the FIR antialias of resample_poly.
    Both filters run at the native 300 Hz, where 49 Hz sits at 0.33x Nyquist
    (stable) rather than 0.98x Nyquist as it would after resampling to 100 Hz.
    """
    b_notch, a_notch = sig.iirnotch(w0=50.0, Q=5.0, fs=FS_RAW)
    x = sig.filtfilt(b_notch, a_notch, raw, axis=0)

    sos = sig.butter(4, [1.0, 49.0], btype="bandpass", fs=FS_RAW, output="sos")
    x = sig.sosfiltfilt(sos, x, axis=0)

    x = sig.resample_poly(x, up=FS, down=FS_RAW, axis=0)
    return x.astype(np.float64)


# ------------------------------------------------------------ multitaper PSD
_TAPERS = dpss(WIN, NW, K_TAPERS)               # (K, WIN)
_FREQS = np.fft.rfftfreq(NFFT, d=1.0 / FS)        # (F,)
_ONEF = (_FREQS >= ONEF_BAND[0]) & (_FREQS <= ONEF_BAND[1])


def _multitaper_psd(seg):
    """seg (WIN, 24) -> PSD (24, F) averaged over K orthogonal Slepian tapers."""
    tapered = _TAPERS[:, :, None] * seg[None, :, :]        # (K, WIN, 24)
    spec = np.fft.rfft(tapered, n=NFFT, axis=1)            # (K, F, 24)
    power = (np.abs(spec) ** 2) / FS                       # periodogram per taper
    psd = power.mean(axis=0)                               # (F, 24)
    return np.maximum(psd.T, 1e-12)                        # (24, F), strictly > 0


def _spectral_slope(psd_log):
    """psd_log (24, F) log10 PSD -> (24, 2) [slope, intercept] over the 1/f band."""
    x = _FREQS[_ONEF]
    coeffs = np.empty((N_EEG_CH, 2), dtype=np.float32)
    for c in range(N_EEG_CH):
        slope, intercept = np.polyfit(x, psd_log[c, _ONEF], deg=1)
        coeffs[c] = (slope, intercept)
    return coeffs


# ------------------------------------------------------------- per recording
class EmptyRecording(Exception):
    """Raised for header-only / sub-second EEG exports that carry no usable data."""


def process_recording(eeg_csv, tf_csv, cs_csv):
    eeg_raw = np.atleast_2d(_read_table(eeg_csv))
    if eeg_raw.shape[0] < FS_RAW or eeg_raw.shape[1] < 2 + N_EEG_CH:
        raise EmptyRecording(f"{eeg_raw.shape[0]} raw samples")
    eeg_raw = eeg_raw[:, 2:2 + N_EEG_CH]                   # drop Millis, Hardware
    eeg = _filter_eeg(eeg_raw)                             # (M, 24) @ 100 Hz
    n_seg = len(eeg) // WIN
    if n_seg == 0:
        raise EmptyRecording(f"{len(eeg)} filtered samples (< one 3 s window)")

    tf_all = _read_table(tf_csv)                           # Millis + 6 pose cols
    cs_all = _read_table(cs_csv)                            # Millis, Rating, ChangePerSec
    tf_ms, cs_ms = tf_all[:, 0], cs_all[:, 0]

    eeg_seg, psd_t, psd_raw, coeff = [], [], [], []
    tf_list, pth_list, joy = [], [], []
    last_tf = np.zeros((1, 7), dtype=np.float64)
    last_joy = 0.0

    for k in range(n_seg):
        s = eeg[k * WIN:(k + 1) * WIN]                     # (WIN, 24)
        eeg_seg.append(s.astype(np.float32))

        psd = _multitaper_psd(s)                           # (24, F)
        psd_log = np.log10(psd)
        psd_raw.append(psd.astype(np.float32))
        psd_t.append(psd_log.T.astype(np.float32))         # (F, 24)
        coeff.append(_spectral_slope(psd_log))

        lo, hi = k * WIN_MS, (k + 1) * WIN_MS              # window in Unity ms
        m = (tf_ms >= lo) & (tf_ms < hi)
        rows = tf_all[m] if m.any() else last_tf
        last_tf = rows
        pth_list.append(rows[:, 0:7].astype(np.float32))   # Millis + 6
        tf_list.append(rows[:, 1:7].astype(np.float32))    # 6 pose cols

        cm = (cs_ms >= lo) & (cs_ms < hi)
        last_joy = float(cs_all[cm, 1].max()) if cm.any() else last_joy
        joy.append([last_joy])

    dataset = dict(
        eeg=np.stack(eeg_seg),                             # (n, T, 24)
        psd=np.stack(psd_t),                               # (n, F, 24)
        psd_raw=np.stack(psd_raw),                         # (n, 24, F)
        spectral_coeff=np.stack(coeff),                    # (n, 24, 2)
        freqs=_FREQS.astype(np.float32),                   # (F,)
        tf=tf_list,                                        # list[n] (Li, 6)
        pth=pth_list,                                      # list[n] (Li, 7)
        joy=np.asarray(joy, dtype=np.float32),             # (n, 1)
    )
    return dataset, n_seg


def _find(raw, stem, stream):
    """Locate <stem>_<stream>.{xlsx,xls,csv} under raw; '' if absent."""
    for ext in (".xlsx", ".xls", ".csv"):
        path = os.path.join(raw, f"{stem}_{stream}{ext}")
        if os.path.isfile(path):
            return path
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="datasets/juliete/raw",
                    help="dir of <PID>_<COND>_<stream>.{xlsx,csv} exports (flat)")
    ap.add_argument("--out", default="datasets/juliete/.cache",
                    help="output .npz cache dir consumed by loader.py")
    ap.add_argument("--pattern", default="*", help="glob on <PID>_<COND> stems")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    stems = sorted({
        os.path.basename(p).rsplit("_EEG.", 1)[0]
        for ext in ("xlsx", "xls", "csv")
        for p in glob.glob(os.path.join(args.raw, f"{args.pattern}_EEG.{ext}"))
    })
    print(f"found {len(stems)} EEG recordings under {args.raw}")

    ok = skipped = empty = failed = 0
    for stem in stems:
        eeg = _find(args.raw, stem, "EEG")
        tf = _find(args.raw, stem, "Transforms")
        cs = _find(args.raw, stem, "SubjectiveCs")
        out = os.path.join(args.out, f"{stem}.npz")
        if not (tf and cs):
            print(f"  skip {stem}: missing Transforms/SubjectiveCs")
            skipped += 1
            continue
        if os.path.isfile(out) and not args.overwrite:
            print(f"  skip {stem}: exists")
            skipped += 1
            continue
        try:
            dataset, n_seg = process_recording(eeg, tf, cs)
            np.savez(out, dataset=np.array(dataset, dtype=object))
            print(f"  ok   {stem}: {n_seg} segments  (joy max {dataset['joy'].max():.3f})")
            ok += 1
        except EmptyRecording as exc:
            print(f"  empty {stem}: {exc}")
            empty += 1
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  FAIL {stem}: {exc}")
            failed += 1

    print(f"done: {ok} written, {skipped} skipped, {empty} empty, {failed} failed -> {args.out}")


if __name__ == "__main__":
    main()

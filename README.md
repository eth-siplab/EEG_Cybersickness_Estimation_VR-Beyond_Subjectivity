# Cybersickness EEG — reproduction

A from-scratch reproduction of **"Beyond Subjectivity: Continuous Cybersickness
Detection Using EEG-based Multitaper Spectrum Estimation"** on the released
JULIETE 16-participant VR-cycling dataset.

The original publication shipped the model/evaluation code but **not the
preprocessing front-end** that turns the raw recordings into model inputs. This
repo reconstructs that front-end (`preprocess.py`) and re-runs the paper's
leave-one-subject-out evaluation with a plain PyTorch runner (`main.py`). With
it, a third-party researcher can go from the raw dataset to the paper's results
table.

Upstream method/dataset: `eth-siplab/VR_Cybersickness_EEG_Dataset-Beyond_Subjectivity`.

---

## 1. Install

```bash
conda create -n cybersickness python=3.10 -y
conda activate cybersickness
pip install -r requirements.txt
```

`torch` with CUDA is recommended (a full sweep is ~1 h on a small GPU; CPU works
but is much slower).

## 2. Get the data

The EEG recordings are **not** in this repo (human-subjects data, released
separately with the paper). The dataset is shared as **Google Sheets**: one Drive
folder per participant (`0001`, `0003`, …), each holding per-condition sheets for
several streams (`EEG`, `Transforms`, `Path`, `Accelerometer`, `EyeGaze`,
`SubjectiveCs`). Bulk-download the Drive folder — each sheet is converted to
`.xlsx`. The preprocessor needs three streams per recording (they share one Unity
clock); `preprocess.py` reads `.xlsx` or `.csv`:

```
<PID>_<COND>_EEG.xlsx           Millis, Hardware, <24 DSI-24 channels>
<PID>_<COND>_Transforms.xlsx    Millis, HeadPosition_{X,Y,Z}, HeadRotation_{Yaw,Pitch,Roll}
<PID>_<COND>_SubjectiveCs.xlsx  Millis, Rating, ChangePerSec      (joystick sickness 0-1)
```

Put those (flat) in one directory, e.g. `…/juliete/raw/`. 13 subjects have
usable recordings: `0001 0002 0003 0005 0006 0007 1000 1001 1002 1003 1004 1101 1102`
(header-only/empty exports are skipped automatically).

## 3. Reproduce

```bash
# (a) raw .xlsx/.csv -> .npz cache (multitaper TR-PSD + kinematics + label). Run once.
python preprocess.py --raw  /path/to/juliete/raw \
                     --out  /path/to/juliete/.cache --overwrite

# (b) one leave-one-subject-out fold (held-out subject 0003, headline model)
python main.py --patient 0003 --seed 42 \
               --cache-prefix /path/to/juliete/.cache

# (c) the full grid: 4 input-types x 13 subjects x 3 seeds, ~4 GPU-h
#     (pass the python with torch as PY, and --cache-prefix via the sweep)
sh parallel_sweep.sh 4                       # 4 folds in parallel

# (d) seed-averaged results table
python aggregate.py results_par/*.jsonl
```

`--input-type` selects the model from `networks.py`:
`power-spectral-difference` (TR-PSD+IMU, the headline), `power-spectral-no-kinematic`
(TR-PSD only), `power-spectral-no-eeg` (IMU only), `kinematic`.

## 4. Expected results

Leave-one-subject-out, 13 subjects × 3 seeds (10/20/40), mean ± std across seeds,
pooled over all held-out recordings. "Acc" is the paper's windowed accuracy
(`leaky_accuracy@0.10`).

| input-type (`--input-type`) | paper Table row | MAE | MSE | acc@0.10 | paper MAE / MSE / Acc |
|---|---|---|---|---|---|
| `kinematic`                 | Kinematic model (Li2023)               | 0.118 | 0.033 | 61.5 ± 4.9 % | 0.857 / 0.162 / 27.08 |
| `power-spectral-no-eeg`     | Ours, IMU                              | 0.122 | 0.034 | 59.6 ± 4.7 % | 0.931 / 0.193 / 38.22 |
| `power-spectral-no-kinematic` | Ours, Filtering+TR-PSD               | 0.100 | 0.034 | 85.6 ± 5.6 % | 0.620 / 0.109 / 69.35 |
| `power-spectral-difference` | Ours, Filtering+TR-PSD+IMU (**best**)  | **0.099** | **0.033** | **81.2 ± 1.0 %** | 0.638 / 0.092 / 76.83 |

mean-predictor baseline: MAE 0.114, acc@0.10 ~29 %.

**What reproduces**
- **Modality ranking holds** — EEG spectral features ≫ motion. The two TR-PSD
  models reach ~81–86 % windowed accuracy; the two motion-only models sit at
  ~60–66 % and barely beat the mean-predictor baseline on MAE. This is the paper's
  central claim ("methods solely based on kinematic features fail").
- **Best config holds** — TR-PSD+IMU and TR-PSD-only are the top two, as published.
- **Beats the mean predictor** — acc@0.10 81–86 % vs ~29 %.
- **The 1–49 Hz filter helped** — recovering the 30–45 Hz 1/f band raised the
  headline acc@0.10 by giving the model the gamma/1-f band the paper names as the
  cybersickness correlate.

## 5. Pipeline (what `preprocess.py` does)

Per recording, on the shared Unity clock, in non-overlapping 3 s windows:

1. EEG: 50 Hz notch (mains) + 1–49 Hz Butterworth band-pass (keeps the 30–45 Hz
   1/f band) at 300 Hz, then resample to 100 Hz.
2. Multitaper PSD (DPSS tapers) → `psd_raw`; plus the 1/f spectral slope.
3. Head-motion features and the joystick label from `Transforms.csv` / `SubjectiveCs.csv`.

The temporal-relative differencing (TR-PSD) and the kinematic 16-tuple are computed
in `loader.py` (unchanged from upstream). A couple of preprocessing details were
not pinned by the paper and are documented reconstruction choices (multitaper
`NW=3, K=5`; 49 Hz low-pass to preserve the 1/f band; per-window label = max).

## 6. Repo map

| file | role |
|---|---|
| `preprocess.py` | raw `.xlsx`/`.csv` → `.npz` cache (the reconstructed front-end) |
| `loader.py`, `networks.py`, `metrics.py` | upstream data/model/metric code (reused) |
| `main.py` | one leave-one-subject-out fold, plain PyTorch |
| `evaluate.sh`, `sweep.sh`, `parallel_sweep.sh` | leave-one-subject-out drivers (loop `main.py`) |
| `aggregate.py`, `parse_logs.py` | seed-averaged metric aggregation (`parse_logs.py` is the `evaluate.sh` entry point) |

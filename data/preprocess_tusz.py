import argparse
import os
import pickle
import shutil
from pathlib import Path
from typing import List, Sequence, Tuple

import mne
import numpy as np
from tqdm import tqdm


BIPOLAR_PAIRS = [
    ("EEG FP1-REF", "EEG F7-REF"),
    ("EEG F7-REF",  "EEG T3-REF"),
    ("EEG T3-REF",  "EEG T5-REF"),
    ("EEG T5-REF",  "EEG O1-REF"),
    ("EEG FP2-REF", "EEG F8-REF"),
    ("EEG F8-REF",  "EEG T4-REF"),
    ("EEG T4-REF",  "EEG T6-REF"),
    ("EEG T6-REF",  "EEG O2-REF"),
    ("EEG FP1-REF", "EEG F3-REF"),
    ("EEG F3-REF",  "EEG C3-REF"),
    ("EEG C3-REF",  "EEG P3-REF"),
    ("EEG P3-REF",  "EEG O1-REF"),
    ("EEG FP2-REF", "EEG F4-REF"),
    ("EEG F4-REF",  "EEG C4-REF"),
    ("EEG C4-REF",  "EEG P4-REF"),
    ("EEG P4-REF",  "EEG O2-REF"),
]
FS = 200
WINDOW_SECONDS = 5


def convert_signals(signals: np.ndarray, raw: mne.io.BaseRaw) -> np.ndarray:
    """Re-reference to the 16-channel bipolar montage shared with TUAB/TUEV.

    Robust to REF/LE/AR suffixes by matching electrode bases.
    """

    def norm(name: str) -> str:
        return name.upper().replace("EEG", "").replace(" ", "").replace(".", "")

    ch_names = [norm(n) for n in raw.info["ch_names"]]
    ch_map_full = {n: i for i, n in enumerate(ch_names)}
    ch_map_base = {}
    for i, n in enumerate(ch_names):
        ch_map_base.setdefault(n.split("-")[0], i)

    def idx(target: str) -> int:
        t_norm = norm(target)
        if t_norm in ch_map_full:
            return ch_map_full[t_norm]
        base = t_norm.split("-")[0]
        if base in ch_map_base:
            return ch_map_base[base]
        raise KeyError(target)

    return np.vstack([signals[idx(a)] - signals[idx(b)] for a, b in BIPOLAR_PAIRS])


def load_events(csv_path: Path) -> List[Tuple[float, float, str]]:
    events = []
    with open(csv_path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            _chan, start, end, label = parts[:4]
            try:
                events.append((float(start), float(end), label))
            except ValueError:
                continue
    return events


def mask_from_events(events: Sequence[Tuple[float, float, str]], times: np.ndarray,
                     target_label: str = "seiz") -> np.ndarray:
    mask = np.zeros(times.shape, dtype=bool)
    for start, end, label in events:
        if target_label not in label:
            continue
        start_idx = np.searchsorted(times, start)
        end_idx = np.searchsorted(times, end)
        mask[start_idx:end_idx] = True
    return mask


def windows_from_mask(mask: np.ndarray, window: int) -> List[Tuple[int, int]]:
    windows: List[Tuple[int, int]] = []
    i, n = 0, mask.size
    while i < n:
        if not mask[i]:
            i += 1
            continue
        start = i
        while i < n and mask[i]:
            i += 1
        cur = start
        while cur + window <= i:
            windows.append((cur, cur + window))
            cur += window
    return windows


def sample_background(mask: np.ndarray, window: int, max_samples: int) -> List[Tuple[int, int]]:
    windows = windows_from_mask(~mask, window)
    if len(windows) > max_samples:
        idx = np.random.choice(len(windows), size=max_samples, replace=False)
        return [windows[i] for i in idx]
    return windows


def save_sample(signal: np.ndarray, label: int, out_path: Path) -> None:
    with open(out_path, "wb") as f:
        pickle.dump({"signal": signal, "label": label}, f)


def process_edf(edf_path: Path, out_dir: Path, bg_ratio: float) -> int:
    csv_path = edf_path.with_suffix(".csv_bi")
    if not csv_path.exists():
        return 0

    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
        raw.resample(FS)
        raw.filter(l_freq=0.3, h_freq=75.0)
        raw.notch_filter(60.0)
        signals = convert_signals(raw.get_data(units="uV"), raw)
    except (ValueError, KeyError):
        return 0

    times = raw.times
    window = FS * WINDOW_SECONDS

    seiz_mask = mask_from_events(load_events(csv_path), times, target_label="seiz")
    seiz_windows = windows_from_mask(seiz_mask, window)
    bg_windows = sample_background(seiz_mask, window, max_samples=int(len(seiz_windows) * bg_ratio + 1))

    count = 0
    stem = edf_path.stem
    for i, (s, e) in enumerate(seiz_windows):
        save_sample(signals[:, s:e], 1, out_dir / f"{stem}_seiz_{i}.pkl")
        count += 1
    for i, (s, e) in enumerate(bg_windows):
        save_sample(signals[:, s:e], 0, out_dir / f"{stem}_bck_{i}.pkl")
        count += 1
    return count


def process_split(base_dir, out_dir: Path, bg_ratio: float) -> int:
    edf_files = list(Path(base_dir).rglob("*.edf"))
    total = 0
    for edf_path in tqdm(edf_files, desc=f"Processing {Path(base_dir).name}", ncols=100):
        total += process_edf(edf_path, out_dir, bg_ratio=bg_ratio)
    return total


def main():
    parser = argparse.ArgumentParser(description="Preprocess TUSZ EDF + CSV_BI data into pickles.")
    parser.add_argument("--input-dir", required=True,
                        help="Root of TUSZ EDF data (contains 'train/' and 'eval/' subdirs).")
    parser.add_argument("--output-dir", required=True,
                        help="Directory under which train/, val/, test/ pickle splits are written.")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Fraction of TUSZ train subjects held out for validation.")
    parser.add_argument("--bg-ratio", type=float, default=1.0,
                        help="Max background windows per seizure window, per file.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)

    root = args.input_dir
    out = Path(args.output_dir)
    interim = out / "_interim"
    interim_train = interim / "train"
    interim_eval = interim / "eval"
    interim_train.mkdir(parents=True, exist_ok=True)
    interim_eval.mkdir(parents=True, exist_ok=True)

    process_split(os.path.join(root, "train"), interim_train, bg_ratio=args.bg_ratio)
    process_split(os.path.join(root, "eval"),  interim_eval,  bg_ratio=args.bg_ratio)

    # Subject-level 80/20 split of the train pool.
    train_files = os.listdir(interim_train)
    all_train_sub = sorted({f.split("_")[0] for f in train_files})
    cut = int(len(all_train_sub) * (1.0 - args.val_ratio))
    train_sub, val_sub = set(all_train_sub[:cut]), set(all_train_sub[cut:])
    print(f"train subjects: {len(train_sub)}, val subjects: {len(val_sub)}")

    val_files = [f for f in train_files if f.split("_")[0] in val_sub]
    train_files = [f for f in train_files if f.split("_")[0] in train_sub]
    test_files = os.listdir(interim_eval)

    for name in ("train", "val", "test"):
        (out / name).mkdir(parents=True, exist_ok=True)

    for f in tqdm(train_files, desc="train"):
        shutil.copy(interim_train / f, out / "train" / f)
    for f in tqdm(val_files, desc="val"):
        shutil.copy(interim_train / f, out / "val" / f)
    for f in tqdm(test_files, desc="test"):
        shutil.copy(interim_eval / f, out / "test" / f)

    print("Done!")


if __name__ == "__main__":
    main()

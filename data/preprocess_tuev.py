import argparse
import os
import pickle
import shutil
from pathlib import Path

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
EVENT_WINDOW_SECONDS = 5


def build_events(signals, times, event_data):
    num_events = event_data.shape[0]
    num_chan = signals.shape[0]
    features = np.zeros([num_events, num_chan, FS * EVENT_WINDOW_SECONDS])
    offending_channel = np.zeros([num_events, 1])
    labels = np.zeros([num_events, 1])
    offset = signals.shape[1]
    signals = np.concatenate([signals, signals, signals], axis=1)
    for i in range(num_events):
        start = np.where(times >= event_data[i, 1])[0][0]
        end = np.where(times >= event_data[i, 2])[0][0]
        features[i, :] = signals[
            :, offset + start - 2 * FS : offset + end + 2 * FS
        ]
        offending_channel[i, :] = int(event_data[i, 0])
        labels[i, :] = int(event_data[i, 3])
    return features, offending_channel, labels


def convert_signals(signals, raw):
    name_to_idx = {ch: i for i, ch in enumerate(raw.info["ch_names"])}
    return np.vstack([
        signals[name_to_idx[a]] - signals[name_to_idx[b]]
        for a, b in BIPOLAR_PAIRS
    ])


def read_edf(file_name):
    raw = mne.io.read_raw_edf(file_name, preload=True)
    raw.resample(FS)
    raw.filter(l_freq=0.3, h_freq=75)
    raw.notch_filter(60)
    _, times = raw[:]
    signals = raw.get_data(units="uV")
    rec_file = file_name[:-3] + "rec"
    event_data = np.genfromtxt(rec_file, delimiter=",")
    raw.close()
    return signals, times, event_data, raw


def process_split(base_dir, out_dir):
    for dir_name, _sub_dirs, file_list in tqdm(os.walk(base_dir)):
        print(f"Found directory: {dir_name}")
        for fname in file_list:
            if not fname.endswith(".edf"):
                continue
            try:
                signals, times, event_data, raw = read_edf(os.path.join(dir_name, fname))
                signals = convert_signals(signals, raw)
            except (ValueError, KeyError):
                print(f"  skipping (could not parse): {dir_name}/{fname}")
                continue
            event_signals, offending_channels, labels = build_events(signals, times, event_data)
            for idx, (signal, ch, label) in enumerate(zip(event_signals, offending_channels, labels)):
                sample = {"signal": signal, "offending_channel": ch, "label": label}
                with open(os.path.join(out_dir, f"{fname.split('.')[0]}-{idx}.pkl"), "wb") as f:
                    pickle.dump(sample, f)


def main():
    parser = argparse.ArgumentParser(description="Preprocess TUEV EDF + REC data into pickles.")
    parser.add_argument("--input-dir", required=True,
                        help="Root of TUEV v2.0.1 EDF data (contains 'train/' and 'eval/' subdirs).")
    parser.add_argument("--output-dir", required=True,
                        help="Directory under which train/, val/, test/ pickle splits are written.")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Fraction of TUEV train subjects held out for validation.")
    args = parser.parse_args()

    root = args.input_dir
    out = Path(args.output_dir)
    interim = out / "_interim"
    interim_train = interim / "train"
    interim_eval = interim / "eval"
    interim_train.mkdir(parents=True, exist_ok=True)
    interim_eval.mkdir(parents=True, exist_ok=True)

    process_split(os.path.join(root, "train"), str(interim_train))
    process_split(os.path.join(root, "eval"),  str(interim_eval))

    # Subject-level 80/20 split of the train pool.
    train_files = os.listdir(interim_train)
    all_train_sub = sorted({f.split("_")[0] for f in train_files})
    cut = int(len(all_train_sub) * (1.0 - args.val_ratio))
    train_sub, val_sub = all_train_sub[:cut], all_train_sub[cut:]
    print(f"train subjects: {len(train_sub)}, val subjects: {len(val_sub)}")

    val_files = [f for f in train_files if f.split("_")[0] in set(val_sub)]
    train_files = [f for f in train_files if f.split("_")[0] in set(train_sub)]
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

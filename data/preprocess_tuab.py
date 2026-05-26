import argparse
import os
import pickle
from multiprocessing import Pool

import mne


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
WINDOW_SAMPLES = 2000  # 10 s @ 200 Hz


def split_and_dump(params):
    fetch_folder, sub, dump_folder, label, error_log = params
    for file in os.listdir(fetch_folder):
        if sub not in file:
            continue
        print("process", file)
        raw = mne.io.read_raw_edf(os.path.join(fetch_folder, file), preload=True)
        raw.resample(200)
        raw.filter(l_freq=0.3, h_freq=75)
        raw.notch_filter(60)
        ch_name = raw.ch_names
        raw_data = raw.get_data(units="uV")
        channeled_data = raw_data.copy()[: len(BIPOLAR_PAIRS)]
        try:
            for i, (a, b) in enumerate(BIPOLAR_PAIRS):
                channeled_data[i] = raw_data[ch_name.index(a)] - raw_data[ch_name.index(b)]
        except ValueError:
            with open(error_log, "a") as f:
                f.write(file + "\n")
            continue
        for i in range(channeled_data.shape[1] // WINDOW_SAMPLES):
            dump_path = os.path.join(
                dump_folder, file.split(".")[0] + "_" + str(i) + ".pkl"
            )
            window = channeled_data[:, i * WINDOW_SAMPLES : (i + 1) * WINDOW_SAMPLES]
            with open(dump_path, "wb") as f:
                pickle.dump({"X": window, "y": label}, f)


def _list_subjects(folder):
    return sorted({item.split("_")[0] for item in os.listdir(folder)})


def main():
    parser = argparse.ArgumentParser(description="Preprocess TUAB EDF data into pickles.")
    parser.add_argument("--input-dir", required=True,
                        help="Root of TUAB v3.0.1 EDF data "
                             "(contains 'train/{normal,abnormal}' and 'eval/{normal,abnormal}').")
    parser.add_argument("--output-dir", required=True,
                        help="Directory under which train/, val/, test/ pickle splits are written.")
    parser.add_argument("--channel-std", default="01_tcp_ar",
                        help="Channel-configuration subfolder name (default '01_tcp_ar').")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Fraction of TUAB train subjects held out for validation.")
    parser.add_argument("--num-workers", type=int, default=24)
    args = parser.parse_args()

    root = args.input_dir
    out = args.output_dir
    channel_std = args.channel_std

    train_val_abnormal = os.path.join(root, "train", "abnormal", channel_std)
    train_val_normal   = os.path.join(root, "train", "normal",   channel_std)
    test_abnormal      = os.path.join(root, "eval",  "abnormal", channel_std)
    test_normal        = os.path.join(root, "eval",  "normal",   channel_std)

    a_sub = _list_subjects(train_val_abnormal)
    n_sub = _list_subjects(train_val_normal)
    cut_a = int(len(a_sub) * (1.0 - args.val_ratio))
    cut_n = int(len(n_sub) * (1.0 - args.val_ratio))
    train_a, val_a = a_sub[:cut_a], a_sub[cut_a:]
    train_n, val_n = n_sub[:cut_n], n_sub[cut_n:]
    test_a = _list_subjects(test_abnormal)
    test_n = _list_subjects(test_normal)

    print(f"train abnormal subjects: {len(train_a)}, val: {len(val_a)}")
    print(f"train normal   subjects: {len(train_n)}, val: {len(val_n)}")
    print(f"test  abnormal subjects: {len(test_a)}, normal: {len(test_n)}")

    train_dump = os.path.join(out, "train")
    val_dump   = os.path.join(out, "val")
    test_dump  = os.path.join(out, "test")
    for d in (train_dump, val_dump, test_dump):
        os.makedirs(d, exist_ok=True)
    error_log = os.path.join(out, "tuab-process-error-files.txt")

    parameters = []
    for s in train_a: parameters.append([train_val_abnormal, s, train_dump, 1, error_log])
    for s in train_n: parameters.append([train_val_normal,   s, train_dump, 0, error_log])
    for s in val_a:   parameters.append([train_val_abnormal, s, val_dump,   1, error_log])
    for s in val_n:   parameters.append([train_val_normal,   s, val_dump,   0, error_log])
    for s in test_a:  parameters.append([test_abnormal,      s, test_dump,  1, error_log])
    for s in test_n:  parameters.append([test_normal,        s, test_dump,  0, error_log])

    with Pool(processes=args.num_workers) as pool:
        pool.map(split_and_dump, parameters)
    print("Done!")


if __name__ == "__main__":
    main()

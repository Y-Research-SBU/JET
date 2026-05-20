"""Preprocess the TUSZ (TUH Seizure) corpus into 5-second pickle clips.

TUSZ ships per-recording seizure annotations (`.tse` / `.tse_bi`). Each output
pickle contains a 16-channel x 1000-sample window (bipolar montage, resampled
to 200 Hz, 0.3-75 Hz band-pass, 60 Hz notch) and a binary label
(0 = background, 1 = seizure).

NOTE: TUSZ preprocessing is dataset-specific (it depends on which TUSZ
revision and which annotation flavour you use). This script is provided as a
stub — fill in the body to match your TUSZ release before running.
"""

import argparse
from pathlib import Path


def process_tusz_dataset(input_dir: Path, output_dir: Path):
    raise NotImplementedError(
        "Fill in TUSZ preprocessing for your local TUSZ revision; the expected "
        "output format is a directory of pickles with keys "
        "{'signal': (16, 1000) float64 array, 'label': int}, sharded into "
        f"{output_dir}/train, {output_dir}/val, {output_dir}/test."
    )


def main():
    parser = argparse.ArgumentParser(description="Preprocess TUSZ EDF data into pickles.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    process_tusz_dataset(Path(args.input_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()

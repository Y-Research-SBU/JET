<div align="center">

<h1>Let EEG Models Learn EEG</h1>

Yifan Wang<sup>1</sup>, Yijia Ma<sup>2</sup>, Wen Li<sup>2</sup>, Chenyu You<sup>1</sup>

<sup>1</sup>Stony Brook University &nbsp; <sup>2</sup>University of Texas Health Center at Houston

<p>
  <a href="https://arxiv.org/abs/2026.XXXXX">
    <img src="https://img.shields.io/badge/ArXiv-Coming%20Soon-B31B1B?style=flat-square&logo=arxiv" alt="arXiv">
  </a>
  <a href="https://y-research-sbu.github.io/JET/">
    <img src="https://img.shields.io/badge/Project-Website-4285F4?style=flat-square&logo=googlechrome" alt="Project Page">
  </a>
  <!--
  <a href="https://github.com/Y-Research-SBU/JET">
    <img src="https://img.shields.io/badge/GitHub-Code-006400?style=flat-square&logo=github" alt="GitHub">
  </a>
  <a href="https://github.com/Y-Research-SBU/JET">
    <img src="https://img.shields.io/badge/Hugging%20Face-Coming%20Soon-F9A825?style=flat-square&logo=huggingface" alt="Hugging Face">
  </a>
  -->
</p>

</div>

---

<div align="center">
<img src="docs/figures/main.svg" width="99%">
</div>

## Installation

```bash
conda create -n jet python=3.10 -y
conda activate jet
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Data Preprocessing

JET trains on three corpora from the [Temple University Hospital EEG project](https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml). Each dataset must be requested and downloaded with the TUH credentials.

| Dataset | Source                                                                                                                                          | Notes                                |
|---------|-------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------|
| TUAB    | [`tuh_eeg_abnormal`](https://isip.piconepress.com/projects/nedc/data/tuh_eeg/tuh_eeg_abnormal/)                                    | normal vs. abnormal recordings       |
| TUEV    | [`tuh_eeg_events`](https://isip.piconepress.com/projects/nedc/data/tuh_eeg/tuh_eeg_events/)                                        | 6-class EEG events (.edf + .rec)     |
| TUSZ    | [`tuh_eeg_seizure`](https://isip.piconepress.com/projects/nedc/data/tuh_eeg/tuh_eeg_seizure/)                                      | background vs. seizure (.edf + .tse) |

All raw recordings are resampled to 200 Hz, band-pass-filtered to 0.3–75 Hz, notch-filtered at 60 Hz, and re-referenced to a 16-channel bipolar montage. Subjects in each TUH training set are split 80/20 into train/val (the official eval set is used as test). Run the per-dataset script:

```bash
python data/preprocess_tuab.py \
  --input-dir  /path/to/tuh_eeg_abnormal/edf \
  --output-dir ./datasets/tuab

python data/preprocess_tuev.py \
  --input-dir  /path/to/tuh_eeg_events/edf \
  --output-dir ./datasets/tuev

python data/preprocess_tusz.py \
  --input-dir  /path/to/tuh_eeg_seizure/edf \
  --output-dir ./datasets/tusz
```

The resulting layout is:

```text
datasets/
├── tuab/
│   ├── train/*.pkl    
│   ├── val/*.pkl
│   └── test/*.pkl
├── tuev/
│   ├── train/*.pkl    
│   ├── val/*.pkl
│   └── test/*.pkl
└── tusz/
    ├── train/*.pkl    
    ├── val/*.pkl
    └── test/*.pkl
```

## Training

Train JET on TUAB / TUEV / TUSZ. The scripts use the paper's robust default constraint weights (L<sub>cons</sub>=1, L<sub>tv</sub>=0.1, L<sub>corr</sub>=0.1) and the standard hyperparameters (AdamW, base LR 5e-5, batch size 256, 200 epochs, EMA 0.9999, label-drop 0.1, log-normal time prior with P<sub>mean</sub>=-0.8, P<sub>std</sub>=0.8, t<sub>ε</sub>=5e-2).

```bash
bash scripts/train_tuab.sh /path/to/datasets/tuab ./output/tuab
bash scripts/train_tuev.sh /path/to/datasets/tuev ./output/tuev
bash scripts/train_tusz.sh /path/to/datasets/tusz ./output/tusz
```

Or invoke `train_eeg.py` directly:

```bash
python train_eeg.py \
  --dataset tuab \
  --datasets_dir /path/to/datasets/tuab \
  --output_dir ./output/tuab \
  --model JiT-B/16 \
  --num_eeg_channels 16 --target_length 2000 --eeg_patch_size 200 \
  --batch_size 256 --epochs 200 --blr 5e-5 \
  --loss_type mix --loss_weight_stat 1.0 --loss_weight_tv 0.1 --loss_weight_corr 0.1
```

Training writes a TensorBoard run and `checkpoint-last.pth` under `--output_dir`.

## Inference

Sample from a checkpoint and compute TS-FID. Released checkpoints ship as **weights only**; `inference.py` applies the paper's reported configuration (Heun sampler, 50 steps, CFG = 1.0, log-normal time prior with P<sub>mean</sub>=-0.8 / P<sub>std</sub>=0.8, t<sub>ε</sub>=5e-2, Gaussian noise prior) — only the dataset path, checkpoint path, and output directory are required.

```bash
bash scripts/infer_tuab.sh /path/to/datasets/tuab ./ckpt/jet_tuab ./output/eval_tuab
bash scripts/infer_tuev.sh /path/to/datasets/tuev ./ckpt/jet_tuev ./output/eval_tuev
bash scripts/infer_tusz.sh /path/to/datasets/tusz ./ckpt/jet_tusz ./output/eval_tusz
```

Or invoke `inference.py` directly:

```bash
python inference.py \
  --dataset tuab \
  --datasets_dir /path/to/datasets/tuab \
  --resume ./ckpt/jet_tuab \
  --output_dir ./output/eval_tuab \
  --num_images 0 --gen_bsz 64 \
  --eval_split train --eval_label_mode match_gt
```

Each run writes `eval_batch.npz` (generated traces + matched ground truth + labels) and `metrics.json` (overall + per-class TS-FID) under the output directory.

## Released Checkpoints

| Dataset | Checkpoint |
|---|---|
| TUAB | `ckpt/jet_tuab` |
| TUEV | `ckpt/jet_tuev` |
| TUSZ | `ckpt/jet_tusz` |

All checkpoints use the `JiT-B/16` backbone, 16 EEG channels, and Heun sampling (50 steps).

## Repository Layout

```text
JET/
  train_eeg.py        # training entry
  inference.py        # sampling + TS-FID evaluation
  denoiser.py         # flow-matching wrapper + principled-constraint losses
  engine_eeg.py       # per-epoch training loop
  models/             # raw-ViT backbone
  data/               # TUAB / TUEV / TUSZ loaders + TS-FID metric
  util/               # logging + misc helpers
  scripts/            # local (non-slurm) training & inference shell scripts
  ckpt/               # released checkpoints (kept locally; distributed separately)
  docs/               # project page + figures
```

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{wang2026jet,
  title   = {Let EEG Models Learn EEG},
  author  = {Wang, Yifan and Ma, Yijia and Li, Wen and You, Chenyu},
  journal = {ICML},
  year    = {2026}
}
```

## License

This project is released under the [MIT License](LICENSE).

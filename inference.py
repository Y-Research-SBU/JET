"""Generate EEG samples from a JET checkpoint and compute TS-FID."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from denoiser import Denoiser
from data.data import build_base_dataset, get_label_names
from data.metrics import collect_ground_truth_batch, compute_ts_fid_streaming


_TARGET_LENGTH = {"tuab": 2000, "tuev": 1000, "tusz": 1000}
_NUM_CLASSES = {"tuab": 2, "tuev": 6, "tusz": 2}


def get_args_parser():
    parser = argparse.ArgumentParser("JET inference (sampling + TS-FID)", add_help=False)

    parser.add_argument("--dataset", required=True, choices=["tuab", "tuev", "tusz"])
    parser.add_argument("--datasets_dir", type=str, required=True,
                        help="Root folder of the EEG dataset (used to sample reference traces).")
    parser.add_argument("--resume", type=str, required=True,
                        help="Path to a .pth checkpoint or a directory containing checkpoint-last.pth.")
    parser.add_argument("--output_dir", default="./output/eval")
    parser.add_argument("--eval_split", default="test", choices=["train", "val", "test"])

    parser.add_argument("--model", default="JiT-B/16")
    parser.add_argument("--num_eeg_channels", type=int, default=16)
    parser.add_argument("--eeg_patch_size", type=int, default=200)
    parser.add_argument("--target_length", type=int, default=None,
                        help="Defaults to 2000 (TUAB) / 1000 (TUEV, TUSZ).")
    parser.add_argument("--attn_dropout", type=float, default=0.0)
    parser.add_argument("--proj_dropout", type=float, default=0.0)

    parser.add_argument("--noise_type", default="gs", choices=["gs", "zero"])
    parser.add_argument("--noise_scale", type=float, default=1.0)
    parser.add_argument("--t_eps", type=float, default=5e-2)
    parser.add_argument("--P_mean", type=float, default=-0.8)
    parser.add_argument("--P_std", type=float, default=0.8)
    parser.add_argument("--label_drop_prob", type=float, default=0.1)
    parser.add_argument("--ema_decay1", type=float, default=0.9999)
    parser.add_argument("--ema_decay2", type=float, default=0.9996)

    parser.add_argument("--loss_type", default="mix", choices=["l1", "l2", "mix"])
    parser.add_argument("--loss_weight_stft", type=float, default=0.0)
    parser.add_argument("--loss_weight_stat", type=float, default=1.0)
    parser.add_argument("--loss_weight_tv", type=float, default=0.1)
    parser.add_argument("--loss_weight_corr", type=float, default=0.1)

    parser.add_argument("--sampling_method", default="heun", choices=["euler", "heun"])
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--interval_min", type=float, default=0.0)
    parser.add_argument("--interval_max", type=float, default=1.0)

    parser.add_argument("--num_images", type=int, default=0,
                        help="0 with --eval_label_mode match_gt means 'match the full GT split'.")
    parser.add_argument("--gen_bsz", type=int, default=64)
    parser.add_argument("--eval_label_mode", default="match_gt",
                        choices=["round_robin", "match_gt"])
    parser.add_argument("--fid_feature_bins", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")

    return parser


def _resolve_ckpt_path(resume: str) -> Path:
    p = Path(resume)
    if p.is_file():
        return p
    candidate = p / "checkpoint-last.pth"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Checkpoint not found at {resume}")


def _build_labels(num_images, class_num, mode, gt_dataset):
    if mode == "match_gt":
        hist = {label: 0 for label in range(class_num)}
        for idx in range(len(gt_dataset)):
            _, lbl = gt_dataset[idx]
            hist[int(lbl)] += 1
        labels = []
        for label, count in hist.items():
            labels.extend([label] * count)
        return labels[:num_images] if num_images > 0 else labels
    if num_images <= 0:
        raise ValueError("round_robin mode requires --num_images > 0 "
                         "(set a positive count or switch to --eval_label_mode match_gt).")
    labels = []
    while len(labels) < num_images:
        labels.extend(list(range(class_num)))
    return labels[:num_images]


def main(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = args.dataset.lower()
    args.class_num = _NUM_CLASSES[dataset_name]
    if args.target_length is None:
        args.target_length = _TARGET_LENGTH[dataset_name]
    if args.target_length % args.eeg_patch_size != 0:
        raise ValueError("target_length must be divisible by eeg_patch_size")
    args.eeg_patch_num = args.target_length // args.eeg_patch_size

    print(f"Dataset: {dataset_name}  |  arch: {args.model} / {args.num_eeg_channels}ch / "
          f"length={args.target_length} / patch={args.eeg_patch_size}")
    print(f"Sampling: {args.sampling_method} / steps={args.num_sampling_steps} / cfg={args.cfg} / "
          f"t_eps={args.t_eps} / noise={args.noise_type}")

    model = Denoiser(args).to(device)
    model.eval()
    ckpt_path = _resolve_ckpt_path(args.resume)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.ema_params1 = [ckpt["model_ema1"][n].to(device) for n, _ in model.named_parameters()]
    model.ema_params2 = [ckpt["model_ema2"][n].to(device) for n, _ in model.named_parameters()]
    print(f"Loaded weights from {ckpt_path}")

    gt_dataset = build_base_dataset(dataset_name, args.eval_split, Path(args.datasets_dir))
    label_names = get_label_names(dataset_name)
    print(f"GT split '{args.eval_split}' has {len(gt_dataset)} samples.")

    labels = _build_labels(args.num_images, args.class_num, args.eval_label_mode, gt_dataset)
    print(f"Generating {len(labels)} samples (mode={args.eval_label_mode}).")

    chunk = args.gen_bsz
    generated_chunks = []
    for start in range(0, len(labels), chunk):
        end = min(start + chunk, len(labels))
        labels_t = torch.tensor(labels[start:end], device=device, dtype=torch.long)
        with torch.no_grad():
            outputs = model.generate(labels_t).detach()
        outputs = outputs.reshape(outputs.shape[0], args.num_eeg_channels, -1).cpu()
        generated_chunks.append(outputs)
        print(f"  generated {end}/{len(labels)}")
    generated_eeg = torch.cat(generated_chunks, dim=0)

    gt_traces = collect_ground_truth_batch(
        dataset=gt_dataset,
        labels=labels,
        num_channels=args.num_eeg_channels,
        target_length=args.target_length,
        seed=args.seed,
    )

    np.savez_compressed(
        output_dir / "eval_batch.npz",
        generated=generated_eeg.numpy(),
        labels=np.asarray(labels, dtype=np.int64),
        ground_truth=gt_traces.numpy(),
    )

    fid_overall = compute_ts_fid_streaming(
        generated=generated_eeg, reference=gt_traces,
        chunk_size=args.gen_bsz, feature_bins=args.fid_feature_bins,
    )
    labels_t = torch.tensor(labels, dtype=torch.long)
    per_class = {}
    for label in sorted(set(labels)):
        mask = labels_t == label
        if int(mask.sum().item()) < 2:
            per_class[str(label)] = None
            continue
        fid_val = compute_ts_fid_streaming(
            generated=generated_eeg[mask], reference=gt_traces[mask],
            chunk_size=args.gen_bsz, feature_bins=args.fid_feature_bins,
        )
        per_class[str(label)] = fid_val if (isinstance(fid_val, float) and math.isfinite(fid_val)) else None

    summary = {
        "num_samples": len(labels),
        "label_histogram": {str(l): int((labels_t == l).sum().item()) for l in sorted(set(labels))},
        "label_names": label_names,
        "ts_fid": {
            "overall": fid_overall if (isinstance(fid_overall, float) and math.isfinite(fid_overall)) else None,
            "per_class": per_class,
            "feature_bins": args.fid_feature_bins,
        },
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    if isinstance(fid_overall, float) and math.isfinite(fid_overall):
        print(f"TS-FID: {fid_overall:.6f}")
    print(f"Wrote {output_dir / 'eval_batch.npz'} and {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)

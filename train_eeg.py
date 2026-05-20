"""Train JET on TUAB / TUEV / TUSZ."""

import argparse
import datetime
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from denoiser import Denoiser
from engine_eeg import train_one_epoch
import util.misc as misc

from data.data import build_dataloaders


_TARGET_LENGTH_DEFAULTS = {"tuab": 2000, "tuev": 1000, "tusz": 1000}


def get_args_parser():
    parser = argparse.ArgumentParser("JET training", add_help=False)

    # architecture
    parser.add_argument("--model", default="JiT-B/16", type=str, metavar="MODEL")
    parser.add_argument("--attn_dropout", type=float, default=0.0)
    parser.add_argument("--proj_dropout", type=float, default=0.0)
    parser.add_argument("--eeg_patch_size", type=int, default=200)

    # dataset
    parser.add_argument("--dataset", default="tuab", choices=["tuab", "tuev", "tusz"])
    parser.add_argument("--datasets_dir", type=str, required=True,
                        help="Root folder of the EEG dataset (expects train/val/test subdirs).")
    parser.add_argument("--num_eeg_channels", type=int, default=16)
    parser.add_argument("--target_length", type=int, default=None,
                        help="Length of each EEG segment; defaults to the dataset's recommended value.")

    # training
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--blr", type=float, default=5e-5, help="Base learning rate (per 256 samples).")
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--lr_schedule", type=str, default="constant", choices=["constant", "cosine"])
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--ema_decay1", type=float, default=0.9999)
    parser.add_argument("--ema_decay2", type=float, default=0.9996)

    # flow-matching / noise schedule
    parser.add_argument("--P_mean", default=-0.8, type=float)
    parser.add_argument("--P_std", default=0.8, type=float)
    parser.add_argument("--noise_scale", default=1.0, type=float)
    parser.add_argument("--noise_type", default="gs", choices=["gs", "zero"])
    parser.add_argument("--t_eps", default=5e-2, type=float)
    parser.add_argument("--label_drop_prob", default=0.1, type=float)

    # loss
    parser.add_argument("--loss_type", default="mix", choices=["l1", "l2", "mix"])
    parser.add_argument("--loss_weight_stft", type=float, default=0.0)
    parser.add_argument("--loss_weight_stat", type=float, default=0.0)
    parser.add_argument("--loss_weight_tv", type=float, default=0.05)
    parser.add_argument("--loss_weight_corr", type=float, default=0.05)

    # sampling
    parser.add_argument("--sampling_method", default="heun", choices=["euler", "heun"])
    parser.add_argument("--num_sampling_steps", default=50, type=int)
    parser.add_argument("--cfg", default=1.0, type=float)
    parser.add_argument("--interval_min", default=0.0, type=float)
    parser.add_argument("--interval_max", default=1.0, type=float)

    # data loading
    parser.add_argument("--weighted_sampler", action="store_true", dest="weighted_sampler", default=True)
    parser.add_argument("--no_weighted_sampler", action="store_false", dest="weighted_sampler")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--pin_mem", action="store_true", default=True)

    # checkpointing
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--resume", default="", type=str,
                        help="Path to a checkpoint file or a directory containing checkpoint-last.pth.")
    parser.add_argument("--save_last_freq", type=int, default=5)
    parser.add_argument("--log_freq", default=100, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--device", default="cuda")

    # early stopping
    parser.add_argument("--early_stop_patience", type=int, default=0,
                        help="Epochs without improvement before stopping (0 disables).")
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-3)

    return parser


def main(args):
    args.distributed = False
    args.rank = 0
    args.gpu = 0
    args.world_size = 1

    print("Arguments:\n{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    os.makedirs(args.output_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=args.output_dir)

    if args.target_length is None:
        args.target_length = _TARGET_LENGTH_DEFAULTS[args.dataset.lower()]

    if args.target_length % args.eeg_patch_size != 0:
        raise ValueError("target_length must be divisible by eeg_patch_size")
    args.eeg_patch_num = args.target_length // args.eeg_patch_size

    data_loader_dict, num_classes = build_dataloaders(
        dataset_name=args.dataset,
        datasets_dir=args.datasets_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transform=None,
        use_weighted_sampler=args.weighted_sampler,
    )
    args.class_num = num_classes
    data_loader_train = data_loader_dict["train"]

    model = Denoiser(args).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))

    eff_batch_size = args.batch_size
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    print("Actual lr: {:.2e}".format(args.lr))
    print("Effective batch size: %d" % eff_batch_size)

    param_groups = misc.add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    if args.resume:
        resume_path = Path(args.resume)
        ckpt_path = resume_path if resume_path.is_file() else resume_path / "checkpoint-last.pth"
        if ckpt_path.exists():
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["model"])
            model.ema_params1 = [ckpt["model_ema1"][n].to(device) for n, _ in model.named_parameters()]
            model.ema_params2 = [ckpt["model_ema2"][n].to(device) for n, _ in model.named_parameters()]
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "epoch" in ckpt:
                args.start_epoch = ckpt["epoch"] + 1
            print(f"Resumed from {ckpt_path}")
        else:
            print(f"Warning: resume path {args.resume} not found; training from scratch.")
            model.ema_params1 = [p.detach().clone() for p in model.parameters()]
            model.ema_params2 = [p.detach().clone() for p in model.parameters()]
    else:
        model.ema_params1 = [p.detach().clone() for p in model.parameters()]
        model.ema_params2 = [p.detach().clone() for p in model.parameters()]

    early_stop_enabled = args.early_stop_patience is not None and args.early_stop_patience > 0
    best_loss = float("inf")
    epochs_since_improvement = 0

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_loss = train_one_epoch(
            model, model, data_loader_train, optimizer, device, epoch,
            log_writer=log_writer, args=args,
        )

        if epoch_loss is not None and early_stop_enabled:
            if epoch_loss + args.early_stop_min_delta < best_loss:
                best_loss = epoch_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(args=args, model_without_ddp=model, optimizer=optimizer,
                            epoch=epoch, epoch_name="last")

        if log_writer is not None:
            log_writer.flush()

        if early_stop_enabled and epochs_since_improvement >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch} (best loss {best_loss:.6f})")
            break

    print("Training time:", str(datetime.timedelta(seconds=int(time.time() - start_time))))


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

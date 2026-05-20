"""TS-FID metric for EEG generation.

Computes a Fréchet distance between generated and reference EEG batches in a
compact frequency-domain feature space. Streaming variant keeps memory bounded.
"""

import math
from collections import Counter
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg


def _to_tensor(data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data.clone().detach()
    return torch.as_tensor(data)


def _eeg_to_standard(eeg: torch.Tensor, num_channels: int, target_length: int) -> torch.Tensor:
    """Reshape a raw dataset sample to [num_channels, target_length]."""
    tensor = eeg.clone().float()
    if tensor.shape[0] != num_channels:
        tensor = tensor.view(num_channels, -1)
    elif tensor.dim() > 2:
        tensor = tensor.view(num_channels, -1)
    length = tensor.shape[1]
    if length < target_length:
        tensor = F.pad(tensor, (0, target_length - length))
    elif length > target_length:
        tensor = tensor[:, :target_length]
    return tensor


def _prepare_fid_features(
    batch: torch.Tensor,
    feature_bins: int = 64,
    spatial_bins: Optional[int] = 4,
    max_frequency_ratio: Optional[float] = 0.5,
) -> np.ndarray:
    """Map a batch [N, C, T] -> features [N, D] via FFT magnitude + log1p + standardize."""
    batch = batch.to(torch.float64)
    freq = torch.fft.rfft(batch, dim=-1).abs()

    if max_frequency_ratio is not None:
        ratio = float(max(0.0, min(1.0, max_frequency_ratio)))
        if ratio > 0.0:
            max_freq = max(1, int(freq.shape[-1] * ratio))
            freq = freq[..., :max_freq]

    if feature_bins is not None and feature_bins > 0:
        freq = F.adaptive_avg_pool1d(freq, min(feature_bins, freq.shape[-1]))

    if spatial_bins is not None and spatial_bins > 0 and freq.shape[1] > spatial_bins:
        freq = freq.permute(0, 2, 1)
        freq = F.adaptive_avg_pool1d(freq, spatial_bins)
        freq = freq.permute(0, 2, 1)

    features = freq.reshape(freq.shape[0], -1)
    features = torch.log1p(features)
    mean = features.mean(dim=1, keepdim=True)
    std = features.std(dim=1, keepdim=True).clamp_min(1e-6)
    features = (features - mean) / std
    return features.cpu().numpy()


def _fid_feature_iterator(batch, chunk_size, feature_bins, spatial_bins, max_frequency_ratio):
    total = batch.shape[0]
    for start in range(0, total, chunk_size):
        chunk = batch[start : start + chunk_size]
        if chunk.size(0) == 0:
            continue
        features = _prepare_fid_features(chunk, feature_bins, spatial_bins, max_frequency_ratio)
        yield torch.as_tensor(features, dtype=torch.float64)


def _stats_from_features(iterator):
    total = 0
    sum_vec = None
    sum_outer = None
    for feat in iterator:
        if feat.numel() == 0:
            continue
        if sum_vec is None:
            dim = feat.shape[1]
            sum_vec = torch.zeros(dim, dtype=torch.float64)
            sum_outer = torch.zeros(dim, dim, dtype=torch.float64)
        total += feat.shape[0]
        sum_vec += feat.sum(dim=0)
        sum_outer += feat.T @ feat
    if total < 2:
        return total, None, None
    mean = sum_vec / total
    cov = (sum_outer - torch.outer(mean, mean) * total) / (total - 1)
    return total, mean.cpu().numpy(), cov.cpu().numpy()


def compute_ts_fid_streaming(
    generated: torch.Tensor,
    reference: torch.Tensor,
    chunk_size: int = 512,
    feature_bins: int = 64,
    spatial_bins: int = 4,
    max_frequency_ratio: float = 0.5,
) -> float:
    """Streaming TS-FID — feeds chunks through FFT-features then aggregates Gaussian stats."""
    generated = _to_tensor(generated).to(torch.float32).cpu()
    reference = _to_tensor(reference).to(torch.float32).cpu()
    if generated.shape != reference.shape:
        raise ValueError(f"Generated shape {generated.shape} != reference shape {reference.shape}")
    if generated.size(0) < 2 or reference.size(0) < 2:
        return math.nan

    count_g, mu_g, cov_g = _stats_from_features(
        _fid_feature_iterator(generated, chunk_size, feature_bins, spatial_bins, max_frequency_ratio)
    )
    count_r, mu_r, cov_r = _stats_from_features(
        _fid_feature_iterator(reference, chunk_size, feature_bins, spatial_bins, max_frequency_ratio)
    )
    if count_g < 2 or count_r < 2:
        return math.nan

    diff = mu_g - mu_r
    cov_prod = cov_g.dot(cov_r)
    covmean, _ = linalg.sqrtm(cov_prod, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(cov_g.shape[0]) * 1e-6
        covmean, _ = linalg.sqrtm((cov_g + offset).dot(cov_r + offset), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff.dot(diff) + np.trace(cov_g + cov_r - 2.0 * covmean))
    return fid if math.isfinite(fid) else math.nan


def collect_ground_truth_batch(
    dataset,
    labels: Sequence[int],
    num_channels: int,
    target_length: int,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Sample one GT trace per requested label (without replacement within the GT pool)."""
    if dataset is None or len(dataset) == 0:
        raise ValueError("Ground-truth dataset is required and must be non-empty.")
    label_counts = Counter(int(lbl) for lbl in labels)

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()

    collected = {label: [] for label in label_counts}
    required = sum(label_counts.values())
    for idx in indices:
        data, lbl = dataset[idx]
        label_val = int(lbl) if not isinstance(lbl, int) else lbl
        if label_val not in label_counts or len(collected[label_val]) >= label_counts[label_val]:
            continue
        tensor = torch.as_tensor(data, dtype=torch.float32)
        collected[label_val].append(_eeg_to_standard(tensor, num_channels, target_length))
        if sum(len(v) for v in collected.values()) >= required:
            break

    missing = {lbl: label_counts[lbl] - len(v) for lbl, v in collected.items()
               if len(v) < label_counts[lbl]}
    if missing:
        raise RuntimeError(f"Not enough ground-truth samples for labels: {missing}")

    stacks = {lbl: list(reversed(stack)) for lbl, stack in collected.items()}
    ordered = [stacks[int(label)].pop() for label in labels]
    return torch.stack(ordered, dim=0)

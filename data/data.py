import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Callable, Dict, Tuple
import numpy as np

from .tuab_dataset import CustomDataset as TuabDataset
from .tuev_dataset import CustomDataset as TuevDataset
from .tusz_dataset import CustomDataset as TuszDataset


DATASET_REGISTRY = {
    "tuab": {"factory": TuabDataset, "num_classes": 2},
    "tuev": {"factory": TuevDataset, "num_classes": 6},
    "tusz": {"factory": TuszDataset, "num_classes": 2},
}


LABEL_NAME_REGISTRY = {
    "tuab": ["normal", "abnormal"],
    "tuev": ["spsw", "gped", "pled", "eyem", "artf", "bckg"],
    "tusz": ["background", "seizure"],
}


def get_label_names(dataset_name: str):
    return LABEL_NAME_REGISTRY.get(dataset_name.lower())


def _label_to_int(label):
    if isinstance(label, torch.Tensor):
        return int(label.item())
    if isinstance(label, np.ndarray):
        return int(label.reshape(-1)[0])
    if isinstance(label, (list, tuple)):
        return _label_to_int(label[0])
    return int(label)


def _compute_sample_weights(dataset: Dataset, num_classes: int):
    base_dataset = getattr(dataset, "base_dataset", dataset)
    cached = getattr(dataset, "_sample_weights_cache", None)
    if cached is not None:
        return cached

    labels = []
    label_counts = [0 for _ in range(num_classes)]
    for idx in range(len(base_dataset)):
        _, label = base_dataset[idx]
        label_int = _label_to_int(label)
        if label_int < 0 or label_int >= num_classes:
            raise ValueError(f"Label {label_int} is outside [0, {num_classes}).")
        labels.append(label_int)
        label_counts[label_int] += 1

    counts = torch.tensor(label_counts, dtype=torch.float32)
    counts = torch.where(counts > 0, counts, torch.ones_like(counts))
    weights = torch.tensor([1.0 / counts[label] for label in labels], dtype=torch.double)

    dataset._sample_weights_cache = weights
    return weights


class TransformedDataset(Dataset):
    """Apply an optional transform to individual EEG examples."""

    def __init__(self, base_dataset: Dataset, transform: Callable = None):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data, label = self.base_dataset[idx]
        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(int(label), dtype=torch.long)
        if self.transform is not None:
            data = self.transform(data)
        return data, label


def build_base_dataset(dataset_name: str, split: str, datasets_dir: Path):
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unsupported dataset {dataset_name}")

    factory = DATASET_REGISTRY[dataset_name]["factory"]
    datasets_dir = Path(datasets_dir)

    if dataset_name == "tuab":
        return factory(datasets_dir, mode=split)
    if dataset_name == "tuev":
        split_path = datasets_dir / split
        files = sorted([f for f in split_path.iterdir() if f.is_file()])
        return factory(split_path, [f.name for f in files])
    if dataset_name == "tusz":
        return factory(datasets_dir, split=split)

    raise ValueError(f"Unsupported dataset {dataset_name}")


def build_dataloaders(
    dataset_name: str,
    datasets_dir: str,
    batch_size: int,
    num_workers: int,
    transform: Callable = None,
    distributed: bool = False,
    num_tasks: int = 1,
    global_rank: int = 0,
    use_weighted_sampler: bool = False,
) -> Tuple[Dict[str, DataLoader], int]:
    """Construct dataloaders for TUAB/TUEV/TUSZ and return num_classes."""
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset {dataset_name}")

    datasets_dir = Path(datasets_dir)
    entry = DATASET_REGISTRY[dataset_name]
    dataloaders: Dict[str, DataLoader] = {}

    if use_weighted_sampler and distributed and global_rank == 0:
        print("WeightedRandomSampler is disabled under distributed training; falling back to DistributedSampler.")

    for split in ["train", "val", "test"]:
        base_ds = build_base_dataset(dataset_name, split, datasets_dir)
        ds = TransformedDataset(base_ds, transform=transform)
        sampler = None
        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(
                ds, num_replicas=num_tasks, rank=global_rank, shuffle=split == "train"
            )
        elif use_weighted_sampler and split == "train":
            weights = _compute_sample_weights(ds, entry["num_classes"])
            sampler = torch.utils.data.WeightedRandomSampler(
                weights, num_samples=len(weights), replacement=True
            )
            if global_rank == 0:
                print(f"Using WeightedRandomSampler for {dataset_name} {split} (n={len(weights)}).")
        dataloaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(sampler is None) and split == "train",
            drop_last=split == "train",
            num_workers=num_workers,
            pin_memory=True,
            sampler=sampler,
        )

    return dataloaders, entry["num_classes"]

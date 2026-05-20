import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


def to_tensor(array):
    return torch.from_numpy(array).float()


class CustomDataset(Dataset):
    def __init__(self, data_dir, mode="train"):
        super().__init__()
        self.files = [
            os.path.join(data_dir, mode, file)
            for file in os.listdir(os.path.join(data_dir, mode))
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data_dict = pickle.load(open(self.files[idx], "rb"))
        data = data_dict["X"]
        label = data_dict["y"]
        data = data.reshape(16, 10, 200)
        return data / 100.0, label

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        return to_tensor(x_data), to_tensor(y_label)

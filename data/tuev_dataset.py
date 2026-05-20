import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


def to_tensor(array):
    return torch.from_numpy(array).float()


class CustomDataset(Dataset):
    def __init__(self, data_dir, files):
        super().__init__()
        self.data_dir = data_dir
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data_dict = pickle.load(open(os.path.join(self.data_dir, self.files[idx]), "rb"))
        data = data_dict["signal"]
        label = int(data_dict["label"][0] - 1)
        data = data.reshape(16, 5, 200)
        return data / 100.0, label

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        return to_tensor(x_data), to_tensor(y_label).long()

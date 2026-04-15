import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader , Dataset
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation
from NeuralDecoder import MnistNeuralDecoder
from NeuralDecoder import CortexMnistDataset
from torch.utils.data import DataLoader




class CortexMnistDataset(Dataset):
    """Loads a pre-encoded cortex MNIST split from an HDF5 file."""

    def __init__(self, path: str):
        self.path = path
        with h5py.File(path, "r") as f:
            self.length = len(f["images"])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.path, "r") as f:
            spikes_on  = torch.from_numpy(f["spikes_on"][idx][:])
            spikes_off = torch.from_numpy(f["spikes_off"][idx][:])
            image      = torch.from_numpy(f["images"][idx][:])
        return spikes_on, spikes_off, image

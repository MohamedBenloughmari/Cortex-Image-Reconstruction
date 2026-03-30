import torch
import os
from CortexReconstruction import NeuralEncoder
from torchvision import datasets, transforms
from torch.utils.data import Subset
from tqdm import tqdm


class CortextMnist:
    def __init__(self, Tmax: float, dt: float, grid_resolution: int, chunk_size: int = 500):
        self.Tmax = Tmax
        self.dt = dt
        self.grid_resolution = grid_resolution
        self.chunk_size = chunk_size
        self.data_train = None
        self.data_val = None
        self.data_test = None

    def generate_data(self, train_rate: float, val_rate: float, test_rate: float):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        train_idx = int(train_rate * len(full_train_dataset))
        val_idx   = int(val_rate   * len(full_train_dataset))

        self.data_train = Subset(full_train_dataset, range(train_idx))
        self.data_val   = Subset(full_train_dataset, range(train_idx, train_idx + val_idx))

        full_test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        test_idx = int(test_rate * len(full_test_dataset))
        self.data_test = Subset(full_test_dataset, range(test_idx))

    def _encode_split(self, dataset, desc: str):
        for idx in tqdm(range(len(dataset)), desc=desc, unit="sample"):
            encoder = NeuralEncoder(dx=0.3, dy=0.3, dt=self.dt, ds=0.3)
            image, _ = dataset[idx]
            encoder.fit(image, blur_sigma=0.1)
            encoder.simulate_random_walk(sigma=0.07, T=self.Tmax)
            encoder.compute_activations(grid_range=10.0, grid_resolution=self.grid_resolution, type='GLM')
            yield (torch.from_numpy(encoder.spikes_on).to(torch.int8),
                   torch.from_numpy(encoder.spikes_off).to(torch.int8))

    def _encode_and_save_split(self, dataset, desc: str, split_key: str, save_dir: str):
        split_dir = os.path.join(save_dir, split_key)
        os.makedirs(split_dir, exist_ok=True)

        chunk_on, chunk_off = [], []
        chunk_idx = 0

        for on, off in self._encode_split(dataset, desc):
            chunk_on.append(on)
            chunk_off.append(off)

            if len(chunk_on) == self.chunk_size:
                path = os.path.join(split_dir, f"chunk_{chunk_idx:04d}.pt")
                torch.save({"on": torch.stack(chunk_on), "off": torch.stack(chunk_off)}, path)
                print(f"Saved {path}")
                chunk_on, chunk_off = [], []
                chunk_idx += 1

        # Save any remaining samples
        if chunk_on:
            path = os.path.join(split_dir, f"chunk_{chunk_idx:04d}.pt")
            torch.save({"on": torch.stack(chunk_on), "off": torch.stack(chunk_off)}, path)
            print(f"Saved {path}")

    def encode_and_save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        self._encode_and_save_split(self.data_train, "Encoding train", "train", save_dir)
        self._encode_and_save_split(self.data_val,   "Encoding val",   "val",   save_dir)
        self._encode_and_save_split(self.data_test,  "Encoding test",  "test",  save_dir)
        print(f"Done. All chunks saved under '{save_dir}'")


if __name__ == "__main__":
    gen = CortextMnist(Tmax=1, dt=0.02, grid_resolution=40, chunk_size=2000)
    gen.generate_data(train_rate=0.8, val_rate=0.2, test_rate=1)
    gen.encode_and_save("cortex_mnist")


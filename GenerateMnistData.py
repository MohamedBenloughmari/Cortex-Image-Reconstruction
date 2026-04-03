import os
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
from CortexReconstruction import NeuralEncoder


class CortexMnistDataset(Dataset):
    """PyTorch Dataset that loads a pre-encoded cortex MNIST split from disk."""

    def __init__(self, path: str):
        data = torch.load(path)
        self.spikes_on  = data["spikes_on"]   # (N, grid, grid, T)
        self.spikes_off = data["spikes_off"]  # (N, grid, grid, T)
        self.images     = data["images"]      # (N, 1, H, W)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.spikes_on[idx], self.spikes_off[idx], self.images[idx]


class CortexMnistEncoder:
    def __init__(self, Tmax: float, dt: float, grid_resolution: int, save_dir: str = "./cortex_mnist"):
        self.Tmax = Tmax
        self.dt = dt
        self.grid_resolution = grid_resolution
        self.save_dir = save_dir
        self.data_train = None
        self.data_val = None
        self.data_test = None

    def generate_data(self, train_rate: float, val_rate: float, test_rate: float):
        transform = transforms.Compose([
            transforms.ToTensor()
            #transforms.Normalize((0.1307,), (0.3081,))
        ])
        full_train_dataset = datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        train_idx = int(train_rate * len(full_train_dataset))
        val_idx   = int(val_rate   * len(full_train_dataset))

        self.data_train = Subset(full_train_dataset, range(train_idx))
        self.data_val   = Subset(full_train_dataset, range(train_idx, train_idx + val_idx))

        full_test_dataset = datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
        test_idx = int(test_rate * len(full_test_dataset))
        self.data_test = Subset(full_test_dataset, range(test_idx))

    def _encode_and_save(self, dataset, desc: str, filename: str):
        """Encode one split, save it to disk, then free memory."""
        tensors_on  = []
        tensors_off = []
        images      = []

        for idx in tqdm(range(len(dataset)), desc=desc, unit="sample"):
            encoder = NeuralEncoder(dx=0.3, dy=0.3, dt=self.dt, ds=0.3)
            image, label = dataset[idx]
            encoder.fit(image, blur_sigma=0)
            encoder.simulate_random_walk(sigma=0.02, T=self.Tmax)
            encoder.compute_activations(
                grid_range=10.0, grid_resolution=self.grid_resolution, type='GLM'
            )
            tensors_on.append(torch.from_numpy(encoder.spikes_on).to(torch.int8))
            tensors_off.append(torch.from_numpy(encoder.spikes_off).to(torch.int8))
            images.append(image)

        os.makedirs(self.save_dir, exist_ok=True)
        save_path = os.path.join(self.save_dir, filename)
        torch.save({
            "spikes_on":  torch.stack(tensors_on),
            "spikes_off": torch.stack(tensors_off),
            "images":     torch.stack(images),
        }, save_path)
        print(f"Saved {len(images)} samples to '{save_path}'")

    def encode(self):
        self._encode_and_save(self.data_train, "Encoding train", "train.pt")
        self._encode_and_save(self.data_val,   "Encoding val",   "val.pt")
        self._encode_and_save(self.data_test,  "Encoding test",  "test.pt")


# --- Usage: encoding ---
if __name__ == "__main__":
    encoder = CortexMnistEncoder(Tmax=2, dt=0.1, grid_resolution=70, save_dir="./cortex_mnist")
    encoder.generate_data(train_rate=0.8, val_rate=0.2, test_rate=1)
    encoder.encode()

# --- Usage: training ---
# from torch.utils.data import DataLoader
#
# train_dataset = CortexMnistDataset("./cortex_mnist/train.pt")
# val_dataset   = CortexMnistDataset("./cortex_mnist/val.pt")
# test_dataset  = CortexMnistDataset("./cortex_mnist/test.pt")
#
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)
import torch
from CortexReconstruction import NeuralEncoder
from torchvision import datasets, transforms
from torch.utils.data import Subset
from tqdm import tqdm


class CortextMnist:
    def __init__(self, Tmax: float, dt: float, grid_resolution: int):
        self.Tmax = Tmax
        self.dt = dt
        self.grid_resolution = grid_resolution
        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.train_tensors_on = []
        self.val_tensors_on = []
        self.test_tensors_on = []
        self.train_tensors_off = []
        self.val_tensors_off = []
        self.test_tensors_off = []

    def generate_data(self, train_rate: float, val_rate: float, test_rate: float):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        full_train_dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )
        train_idx = int(train_rate * len(full_train_dataset))
        val_idx = int(val_rate * len(full_train_dataset))

        self.data_train = Subset(full_train_dataset, range(train_idx))
        self.data_val = Subset(full_train_dataset, range(train_idx, train_idx + val_idx))

        full_test_dataset = datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )
        test_idx = int(test_rate * len(full_test_dataset))
        self.data_test = Subset(full_test_dataset, range(test_idx))

    def encode(self):
        for idx in tqdm(range(len(self.data_train)), desc="Processing train data", unit="sample"):
            encoder = NeuralEncoder(dx=0.3, dy=0.3, dt=self.dt, ds=0.3)
            image, _ = self.data_train[idx]
            encoder.fit(image, blur_sigma=0.1)
            encoder.simulate_random_walk(sigma=0.07, T=self.Tmax)
            encoder.compute_activations(grid_range=10.0, grid_resolution=self.grid_resolution, type='GLM')
            self.train_tensors_on.append(torch.from_numpy(encoder.spikes_on))
            self.train_tensors_off.append(torch.from_numpy(encoder.spikes_off))

        for idx in tqdm(range(len(self.data_val)), desc="Processing val data", unit="sample"):
            encoder = NeuralEncoder(dx=0.3, dy=0.3, dt=self.dt, ds=0.3)
            image, _ = self.data_val[idx]
            encoder.fit(image, blur_sigma=0.1)
            encoder.simulate_random_walk(sigma=0.07, T=self.Tmax)
            encoder.compute_activations(grid_range=10.0, grid_resolution=self.grid_resolution, type='GLM')
            self.val_tensors_on.append(torch.from_numpy(encoder.spikes_on).to(torch.int8))
            self.val_tensors_off.append(torch.from_numpy(encoder.spikes_off).to(torch.int8))

        for idx in tqdm(range(len(self.data_test)), desc="Processing test data", unit="sample"):
            encoder = NeuralEncoder(dx=0.3, dy=0.3, dt=self.dt, ds=0.3)
            image, _ = self.data_test[idx]
            encoder.fit(image, blur_sigma=0.1)
            encoder.simulate_random_walk(sigma=0.07, T=self.Tmax)
            encoder.compute_activations(grid_range=10.0, grid_resolution=self.grid_resolution, type='GLM')
            self.test_tensors_on.append(torch.from_numpy(encoder.spikes_on))
            self.test_tensors_off.append(torch.from_numpy(encoder.spikes_off))

    def save(self, path: str):
        data = {
            "train": {
                "on":  torch.stack(self.train_tensors_on),
                "off": torch.stack(self.train_tensors_off),
            },
            "val": {
                "on":  torch.stack(self.val_tensors_on),
                "off": torch.stack(self.val_tensors_off),
            },
            "test": {
                "on":  torch.stack(self.test_tensors_on),
                "off": torch.stack(self.test_tensors_off),
            },
        }
        torch.save(data, path)
        print(f"Saved encoded tensors to '{path}'")


if __name__ == "__main__":
    gen = CortextMnist(Tmax=1, dt=0.02, grid_resolution=40)
    gen.generate_data(train_rate=0.8, val_rate=0.2, test_rate=1)
    gen.encode()
    gen.save("cortex_mnist.pt")
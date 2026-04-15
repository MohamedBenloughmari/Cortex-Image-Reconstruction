import os
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
from CortexReconstructionMnist import NeuralEncoder
import numpy as np
#Parameters






class CortexMnistDataset(Dataset):
    """PyTorch Dataset that loads a pre-encoded cortex MNIST split from disk."""

    def __init__(self, path: str):
        data = torch.load(path)
        self.spikes_on  = data["spikes_on"]   
        self.spikes_off = data["spikes_off"]  
        self.images     = data["images"]      

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.spikes_on[idx], self.spikes_off[idx], self.images[idx]


class CortexMnistEncoder:
    def __init__(self, grid_resolution: int, save_dir: str = "./cortex_mnist"):
        self.grid_resolution = grid_resolution
        self.save_dir = save_dir
        self.data_train = None
        self.data_val = None
        self.data_test = None

    def generate_data(self):
        os.makedirs(self.save_dir,exist_ok=True)
        transform = transforms.Compose([
            transforms.ToTensor()
            #transforms.Normalize((0.1307,), (0.3081,))
        ])
        full_dataset = datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )

        self.data = Subset(full_dataset, range(20))
        images=[]
        for image ,_ in tqdm(self.data):
            images.append(image)
        images=torch.stack(images)
        torch.save(images,os.path.join(self.save_dir,"mnist20.pt"))

    def _encode_and_save(self, dataset,type : str, desc: str, filename: str):
        """Encode one split, save it to disk, then free memory."""
        tensors_on  = []
        tensors_off = []
        images      = []

        Tmax_values=np.linspace(0,5,10)[1:]
        dt_values=np.linspace(0,0.1,10)[1:]
        sigma_values=np.linspace(0,0.5,10)[1:]
        
        data_size={
            'train':5000,
            'val' :500,
            'test':500
        }
        n=len(dataset)
        for _ in tqdm(range(data_size[type]), desc=desc, unit="sample"):
            Tmax=np.random.choice(Tmax_values)
            dt=np.random.choice(dt_values)
            sigma=np.random.choice(sigma_values)
            ex=np.random.randint(0,n-1)
            encoder = NeuralEncoder(dx=0.3, dy=0.3, dt=dt, ds=0.3)
            image, _ = dataset[ex]
            encoder.fit(image, blur_sigma=0)
            encoder.simulate_random_walk(sigma=sigma, T=Tmax)
            encoder.compute_activations(
                grid_range=10.0, grid_resolution=self.grid_resolution, type='GLM'
            )
            tensors_on.append(torch.from_numpy(encoder.spikes_on).to(torch.int8))
            tensors_off.append(torch.from_numpy(encoder.spikes_off).to(torch.int8))
            images.append(image)

        os.makedirs(self.save_dir, exist_ok=True)
        save_path = os.path.join(self.save_dir, filename)
        torch.save({
            "spikes_on":  torch.nested.nested_tensor(tensors_on,layout=torch.jagged),
            "spikes_off": torch.nested.nested_tensor(tensors_off,layout=torch.jagged),
            "images":     torch.nested.nested_tensor(images,layout=torch.jagged),
        }, save_path)
        print(f"Saved {len(images)} samples to '{save_path}'")

    def encode(self):

        self._encode_and_save(self.data, type='train',desc="Encoding train",filename="train.pt")
        self._encode_and_save(self.data,type='val',desc= "Encoding val",filename="val.pt")
        self._encode_and_save(self.data,type='test',desc="Encoding test",filename="test.pt")


if __name__ == "__main__":
    encoder = CortexMnistEncoder(grid_resolution=40, save_dir="./cortex_mnist")
    encoder.generate_data()
    encoder.encode()

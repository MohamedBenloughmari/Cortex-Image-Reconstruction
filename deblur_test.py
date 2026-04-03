import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image, ImageFilter

from ConvLSTM import ConvLSTM
from torch import nn


# ─────────────────────────────────────────────────────────────────────────────
#  Model & Dataset
# ─────────────────────────────────────────────────────────────────────────────

class NeuralDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.convlstm = ConvLSTM(
            img_size=(70, 70),
            input_dim=1,
            hidden_dim=3,
            kernel_size=(3, 3),
            batch_first=True,
            bidirectional=False,
            return_sequence=False)
        self.conv1 = nn.Conv2d(in_channels=3,  out_channels=16, kernel_size=31, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=1,  kernel_size=13, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, last_state, _ = self.convlstm(x)
        x = self.conv1(last_state[0])
        x = self.conv2(x)
        return x


class CortexMnistDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        super().__init__()
        data = torch.load(path, mmap=True)
        self.images     = data['images'].unsqueeze(2)
        self.spikes_on  = data['spikes_on'].unsqueeze(2)
        self.spikes_off = data['spikes_off'].unsqueeze(2)

    def __len__(self):
        return len(self.spikes_on)

    def __getitem__(self, idx):
        return (self.spikes_on[idx].clone(),
                self.spikes_off[idx].clone(),
                self.images[idx].clone())


# ─────────────────────────────────────────────────────────────────────────────
#  Deblur
# ─────────────────────────────────────────────────────────────────────────────

class DeblurMethods:

    @staticmethod
    def _to_uint8(image) -> np.ndarray:
        arr = image.squeeze().cpu().numpy() if isinstance(image, torch.Tensor) else np.squeeze(image)
        return (np.clip(arr, 0, 1) * 255).astype(np.uint8)

    @staticmethod
    def pillow_unsharp(image, radius: float = 3.0,
                       percent: int = 300, threshold: int = 20,
                       clip_threshold: float = 0.4) -> np.ndarray:
        img_u8  = DeblurMethods._to_uint8(image)
        pil_img = Image.fromarray(img_u8, mode='L')
        sharp   = pil_img.filter(
            ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold)
        )
        result = np.clip(np.array(sharp).astype(np.float32) / 255.0, 0, 1)

        # Push dim background pixels to black
        result[result < clip_threshold] = 0.0

        return result
# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

    test_dataset = CortexMnistDataset('cortex_mnist/test.pt')
    test_loader  = DataLoader(test_dataset, batch_size=50, shuffle=False)

    model = NeuralDecoder()
    model.load_state_dict(torch.load('NeuralDecoder.pth', map_location='cpu'))
    model.to(device)
    model.eval()
    idx=13
    spikes_on, _, images = next(iter(test_loader))
    spike          = spikes_on[idx].float().to(device)
    original_image = images[idx].squeeze().numpy()

    with torch.no_grad():
        reconstructed    = model(spike.unsqueeze(0)).squeeze().cpu()
        reconstructed_np = np.clip(reconstructed.numpy(), 0, 1)

    db = DeblurMethods()
    results = {
        'Original':              original_image,
        'NeuralDecoder\nOutput': reconstructed_np,
        'UnsharpMask\n(Pillow)': db.pillow_unsharp(reconstructed),
    }

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.5))

    for ax, (title, img) in zip(axes, results.items()):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.axis('off')

    plt.suptitle('Deblur Comparison', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('deblur_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Original shape     : {original_image.shape}")
    print(f"Reconstructed shape: {reconstructed_np.shape}")
    print("Saved → deblur_comparison.png")


if __name__ == '__main__':
    main()
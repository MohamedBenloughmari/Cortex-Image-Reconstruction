import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image, ImageFilter

from ConvLSTM import ConvLSTM
from torch import nn
from NeuralDecoder import MnistNeuralDecoder, CortexMnistDataset


class DeblurMethods:

    @staticmethod
    def _to_uint8(image) -> np.ndarray:
        arr = image.squeeze().cpu().numpy() if isinstance(image, torch.Tensor) else np.squeeze(image)
        return (np.clip(arr, 0, 1) * 255).astype(np.uint8)

    @staticmethod
    def pillow_unsharp(image, radius=3.0, percent=300, threshold=20, clip_threshold=0.4):
        img_u8  = DeblurMethods._to_uint8(image)
        pil_img = Image.fromarray(img_u8, mode='L')
        sharp   = pil_img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))
        result  = np.clip(np.array(sharp).astype(np.float32) / 255.0, 0, 1)
        result[result < clip_threshold] = 0.0
        return result


def plot(idx, spikes_on, images, model, device, axes):
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

    for ax, (title, img) in zip(axes, results.items()):
        ax.clear()
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.axis('off')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

    test_dataset = CortexMnistDataset('cortex_mnist/test.pt')
    test_loader  = DataLoader(test_dataset, batch_size=50, shuffle=False)

    model = MnistNeuralDecoder(grid_size=40)
    model.load_state_dict(torch.load('NeuralDecoder.pth', map_location='cpu'), strict=False)
    model.to(device)
    model.eval()

    spikes_on, _, images = next(iter(test_loader))
    batch_size = len(images)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))
    plt.subplots_adjust(bottom=0.2)
    plt.suptitle('Deblur Comparison', fontsize=12, fontweight='bold')

    idx = [1]  # mutable so the callback can update it
    plot(idx[0], spikes_on, images, model, device, axes)

    ax_btn = fig.add_axes([0.4, 0.05, 0.2, 0.08])
    from matplotlib.widgets import Button
    btn = Button(ax_btn, 'Refresh')

    def on_refresh(_):
        idx[0] = random.randint(0, batch_size - 1)
        plot(idx[0], spikes_on, images, model, device, axes)
        fig.canvas.draw()

    btn.on_clicked(on_refresh)

    plt.savefig('deblur_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
    
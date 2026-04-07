import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Button
from NeuralDecoder import MnistNeuralDecoder
from NeuralDecoder import CortexMnistDataset
from torch.utils.data import DataLoader

if __name__ == '__main__':
    test_dataset  = CortexMnistDataset('cortex_mnist/test.pt')
    test_loader  = DataLoader(test_dataset,  batch_size=500, shuffle=False, num_workers=0)
    spikes_on, _, images = next(iter(test_loader))
    best_model = MnistNeuralDecoder(grid_size=40)
    best_model_state = torch.load('/Users/mohamed/Cortex-Image-Reconstruction/NeuralDecoder10.pth')
    best_model.load_state_dict(best_model_state)
    best_model.eval()

    def get_random_idx():
        return np.random.randint(0, len(images))

    def plot(idx):
        spike = spikes_on[idx].float()
        img = images[idx]
        with torch.no_grad():
            constructed = best_model(spike.unsqueeze(0))

        ax1.clear()
        ax2.clear()
        ax1.imshow(img.squeeze().numpy(), cmap='gray')
        ax1.set_title('Original')
        ax1.axis('off')
        ax2.imshow(constructed.squeeze().numpy(), cmap='gray')
        ax2.set_title('Reconstructed')
        ax2.axis('off')
        fig.suptitle(f'Sample idx: {idx}', fontsize=12)
        fig.canvas.draw()

    def refresh(event):
        plot(get_random_idx())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    plt.subplots_adjust(bottom=0.15)

    ax_btn = plt.axes([0.4, 0.02, 0.2, 0.07])
    btn = Button(ax_btn, 'Refresh')
    btn.on_clicked(refresh)

    plot(get_random_idx())
    plt.show()
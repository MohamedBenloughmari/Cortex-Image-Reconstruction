import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation
from NeuralDecoder import MnistNeuralDecoder
from NeuralDecoder import CortexMnistDataset
from torch.utils.data import DataLoader

if __name__ == '__main__':
    test_dataset  = CortexMnistDataset('cortex_mnist_nested/test.h5')
    test_loader  = DataLoader(test_dataset,  batch_size=500, shuffle=False, num_workers=0)
    spikes_on, _, images = next(iter(test_loader))
    best_model = MnistNeuralDecoder(grid_size=40)
    best_model_state = torch.load('/Users/mohamed/Cortex-Image-Reconstruction/NeuralDecoder10.pth')
    best_model.load_state_dict(best_model_state)
    best_model.eval()
    np.random.seed(25)

    current_idx = [0]  
    anim_holder = [None]

    def get_random_idx():
        return np.random.randint(0, len(images))

    def plot(idx):
        current_idx[0] = idx
        spike = spikes_on[idx].float()
        img = images[idx]
        print(len(spike))
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

    def anim(idx):
        spike = spikes_on[idx].float()   
        img   = images[idx]
        n_frames = spike.shape[0]        

        def update(i):
            spike_slice = spike[0:i + 1]  
            with torch.no_grad():
                constructed = best_model(spike_slice.unsqueeze(0))
            ax1.clear()
            ax2.clear()
            ax1.imshow(img.squeeze().numpy(), cmap='gray')
            ax1.set_title('Original')
            ax1.axis('off')
            ax2.imshow(constructed.squeeze().numpy(), cmap='gray')
            ax2.set_title(f'Reconstructed (t=0:{i + 1})')
            ax2.axis('off')
            fig.suptitle(f'Sample idx: {idx}  —  frame {i + 1}/{n_frames}', fontsize=12)

        anim_holder[0] = FuncAnimation(fig, update, frames=n_frames,
                                       interval=100, repeat=False)
        fig.canvas.draw()

    def refresh(event):
        if anim_holder[0] is not None:
            anim_holder[0].event_source.stop()
            anim_holder[0] = None
        plot(get_random_idx())

    def animate(event):
        if anim_holder[0] is not None:
            anim_holder[0].event_source.stop()
            anim_holder[0] = None
        anim(current_idx[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    plt.subplots_adjust(bottom=0.15)

    ax_btn_refresh = plt.axes([0.28, 0.02, 0.2, 0.07])
    btn_refresh = Button(ax_btn_refresh, 'Refresh')
    btn_refresh.on_clicked(refresh)

    ax_btn_anim = plt.axes([0.52, 0.02, 0.2, 0.07])
    btn_anim = Button(ax_btn_anim, 'Animate')
    btn_anim.on_clicked(animate)

    plot(get_random_idx())
    plt.show()
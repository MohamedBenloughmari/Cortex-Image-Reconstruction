import matplotlib.pyplot as plt 
import torch 
import torch
import numpy as np
import random
from NeuralCortexReconstruction import NeuralEncoder , learn_mnist_dictionary
from torchvision import datasets, transforms
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D  
import pickle
def compute_snr(original, reconstruction):
    if hasattr(original, 'numpy'):
        original = original.detach().cpu().numpy()
    if hasattr(reconstruction, 'numpy'):
        reconstruction = reconstruction.detach().cpu().numpy()
    original_expanded = original[np.newaxis, :, :] 
    noise = reconstruction - original_expanded       
    signal_power = np.mean(original_expanded ** 2)  
    noise_power  = np.mean(noise ** 2, axis=(-2, -1))
    noise_power = np.maximum(noise_power, 1e-10)
    snr_db = 10 * np.log10(signal_power / noise_power)  
    return snr_db


print("Learning MNIST dictionary")

D, mean_offset = learn_mnist_dictionary(n_components=256, n_samples=60000,
                                        alpha=1.0, max_iter=300)
print(f"D shape = {D.shape}, mean_offset = {mean_offset:.3f}")
ds_val = datasets.MNIST(root='./data', train=False, download=True,
                        transform=transforms.ToTensor())
optotype = ds_val[0][0].squeeze(0)
snr_list=[]
for d in tqdm(np.linspace(0.0, 50.0, 11)):
    sim = NeuralEncoder(dx=0.07, dy=0.07, dt=0.066, ds=0.3, D_diff=d)
    sim.fit(optotype, blur_sigma=0.0)
    sim.simulate_random_walk(T=0.750)
    sim.compute_activations(grid_range=30.0, grid_resolution=60)
    sim.decode(
        D, mean_offset=mean_offset,
        n_particles=100, n_samples=100,
        beta=0.001, gamma=0.01,
        adam_iter=50, lr=1e-3,
        anchor_weight=1.0,
        hessian_tau=15,
        hessian_every=1,
        verbose=False
    )
    snr = compute_snr(optotype, sim.S_hat_history)
    snr_list.append(snr)


def snr_plot3d(snr_list, save_path="snr_plot3d.fig.pkl"):
    import matplotlib.pyplot as plt
    import pickle
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from mpl_toolkits.mplot3d import Axes3D

    snr_array = np.array(snr_list)  
    n_d, T = snr_array.shape
    t_vals = np.arange(T)
    d_vals = np.linspace(0.0, 50.0, n_d)  

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection='3d')

    mean_snrs = snr_array.mean(axis=1)
    norm = mcolors.Normalize(vmin=mean_snrs.min(), vmax=mean_snrs.max())
    cmap = cm.viridis

    for d_idx in range(n_d):
        color = cmap(norm(mean_snrs[d_idx]))
        ax.plot(
            t_vals,
            np.full(T, d_vals[d_idx]),  
            snr_array[d_idx],
            color=color,
            linewidth=1.5
        )

    ax.set_xlabel('Time step T')
    ax.set_ylabel('D_diff')
    ax.set_zlabel('SNR (dB)')
    ax.set_title('SNR per Diffusion Coefficient')

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.5, aspect=10, label='Mean SNR (dB)')

    plt.tight_layout()

    with open(save_path, 'wb') as f:
        pickle.dump(fig, f)
    print(f"Figure saved to {save_path}")

    plt.show()

snr_plot3d(snr_list, save_path="snr_plot3d_07.fig.pkl")
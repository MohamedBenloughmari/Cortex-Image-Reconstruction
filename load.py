import matplotlib.pyplot as plt
import pickle

with open("/Users/mohamed/Cortex-Image-Reconstruction/snr_plot3d_07.fig.pkl", 'rb') as f:
    fig = pickle.load(f)

plt.show() 
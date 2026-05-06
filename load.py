import matplotlib.pyplot as plt
import pickle

with open("/Users/mohamed/Cortex-Image-Reconstruction/grid20/snr_plot3d200_01.fig.pkl", 'rb') as f:
    fig = pickle.load(f)

plt.show() 
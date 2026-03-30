import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter


class NeuralEncoder:
    def __init__(self, dx: float, dy: float, dt: float, ds: float = None) -> None:
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.ds = ds
        self.half_n = None
        self.optotype_np = None
        self.n_steps = None
        self.walk = None
        self.activated_indices = None
        self.ganglion_x = None
        self.ganglion_y = None
        self.spikes_on = None 
        self.spikes_off = None
    def fit(self, optotype: torch.Tensor, blur_sigma: float = 1.5) -> None:
        if optotype.dim() == 3:
            optotype = optotype.squeeze(0)
        h, w = optotype.shape
        self.half_n = h // 2
        raw = optotype.numpy()
        self.optotype_np = gaussian_filter(raw, sigma=blur_sigma) if blur_sigma > 0 else raw
    def simulate_random_walk(self, sigma: float, T: float) -> None:
        self.n_steps = int(T / self.dt)
        positions = np.zeros((self.n_steps + 1, 2))
        displacements = np.random.normal(0.0, sigma / self.dx, size=(self.n_steps, 2))
        positions[1:] = np.cumsum(displacements, axis=0)
        self.walk = positions
    def compute_activations(self, grid_range=20.0, grid_resolution=40,
                            activation_threshold: float = 0.05,
                            type: str = 'presence') -> None:
        xg = np.linspace(-grid_range, grid_range, num=grid_resolution)
        yg = np.linspace(-grid_range, grid_range, num=grid_resolution)
        self.ganglion_x = xg
        self.ganglion_y = yg
        H, W = self.optotype_np.shape
        if type == 'presence':
            activated = []
            for t in range(self.n_steps + 1):
                cx, cy = self.walk[t]
                pairs = []
                for gi, gx in enumerate(xg):
                    for gj, gy in enumerate(yg):
                        rel_x = gx - cx
                        rel_y = gy - cy
                        px = int((rel_x / self.dx) + self.half_n)
                        py = int((rel_y / self.dy) + self.half_n)
                        if 0 <= px < W and 0 <= py < H:
                            if self.optotype_np[py, px] > activation_threshold:
                                pairs.append((gj, gi))
                activated.append(pairs)
            self.activated_indices = activated
        elif type == 'GLM':
            if self.ds is None:
                raise ValueError("ds (cone lattice spacing) must be set for GLM mode.")
            lambda0 = 10.0
            lambda1 = 100.0
            sigma_s = 0.5 * self.ds
            sigma_e = 0.203 * self.ds
            sigma2  = sigma_s**2 + sigma_e**2
            n_t  = self.n_steps + 1
            n_gx = len(xg)
            n_gy = len(yg)
            ns   = np.arange(H)
            ms   = np.arange(W)
            px_x = (ns + 0.5 - self.half_n) * self.dx
            px_y = (ms + 0.5 - self.half_n) * self.dy
            S    = self.optotype_np
            spikes_on  = np.zeros((n_t, n_gx, n_gy), dtype=int)
            spikes_off = np.zeros((n_t, n_gx, n_gy), dtype=int)
            for t in range(n_t):
                cx, cy = self.walk[t]
                diff_x = (cx + px_x[np.newaxis, :]) - xg[:, np.newaxis]   
                diff_y = (cy + px_y[np.newaxis, :]) - yg[:, np.newaxis]   
                gx_w = np.exp(-0.5 * diff_x**2 / sigma2)
                gy_w = np.exp(-0.5 * diff_y**2 / sigma2)
                c_raw  = (gx_w @ S @ gy_w.T) / (2.0 * np.pi * sigma2)
                g_norm = np.max(c_raw) if np.max(c_raw) > 0 else 1.0
                c      = c_raw / g_norm
                c_on  = c
                c_off = 1.0 - c
                lam_on  = np.exp(np.log(lambda0) + np.log(lambda1 / lambda0) * c_on)
                lam_off = np.exp(np.log(lambda0) + np.log(lambda1 / lambda0) * c_off)
                spikes_on[t]  = np.random.poisson(lam_on  * self.dt)
                spikes_off[t] = np.random.poisson(lam_off * self.dt)
            self.spikes_on  = spikes_on
            self.spikes_off = spikes_off
        else:
            raise ValueError(f"Unknown type '{type}'. Use 'presence' or 'GLM'.")
    def animate(self, interval=50, save_path=None):
        xg = self.ganglion_x
        yg = self.ganglion_y
        glm_mode = self.spikes_on is not None
        half_x = self.half_n * self.dx
        half_y = self.half_n * self.dy
        X, Y      = np.meshgrid(xg, yg, indexing='ij')
        pts_x     = X.ravel()
        pts_y     = Y.ravel()
        num_cells = len(pts_x)
        fig, (ax_opt, ax_on, ax_off) = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor('white')
        for ax in (ax_opt, ax_on, ax_off):
            ax.set_facecolor('white')
            ax.tick_params(colors='black')
            ax.xaxis.label.set_color('black')
            ax.yaxis.label.set_color('black')
            ax.title.set_color('black')
            ax.set_aspect('equal')  

            for spine in ax.spines.values():
                spine.set_edgecolor('#aaa')
        cx0, cy0 = self.walk[0]
        opt_img = ax_opt.imshow(
            self.optotype_np,
            extent=[cx0 - half_x, cx0 + half_x, cy0 - half_y, cy0 + half_y],
            origin='lower', cmap='gray_r', alpha=0.9, interpolation='bilinear',
        )
        walk_line_opt, = ax_opt.plot([], [], color='steelblue', lw=0.7, alpha=0.5)
        fovea_dot,     = ax_opt.plot([], [], 'b+', markersize=12, markeredgewidth=2)
        ax_opt.set_xlim(xg[0], xg[-1])
        ax_opt.set_ylim(yg[0], yg[-1])
        ax_opt.set_title('Moving optotype')
        ax_opt.set_xlabel('x (deg)')
        ax_opt.set_ylabel('y (deg)')
        time_text = ax_opt.text(0.02, 0.96, '', transform=ax_opt.transAxes,
                                color='black', fontsize=9, va='top')

        on_scatter = ax_on.scatter(pts_x, pts_y,
                                   c=np.zeros(num_cells),
                                   cmap='YlOrRd', vmin=0, vmax=1,
                                   s=18, edgecolors='none', zorder=3)
        walk_line_on, = ax_on.plot([], [], color='steelblue', lw=0.7, alpha=0.4)
        ax_on.set_xlim(xg[0], xg[-1])
        ax_on.set_ylim(yg[0], yg[-1])
        ax_on.set_title('ON cells – spikes')
        ax_on.set_xlabel('x (deg)')
        ax_on.set_ylabel('y (deg)')
        #plt.colorbar(on_scatter, ax=ax_on, label='normalised spike count')

        off_scatter = ax_off.scatter(pts_x, pts_y,
                                     c=np.zeros(num_cells),
                                     cmap='YlGnBu', vmin=0, vmax=1,
                                     s=18, edgecolors='none', zorder=3)
        walk_line_off, = ax_off.plot([], [], color='steelblue', lw=0.7, alpha=0.4)
        ax_off.set_xlim(xg[0], xg[-1])
        ax_off.set_ylim(yg[0], yg[-1])
        ax_off.set_title('OFF cells – spikes')
        ax_off.set_xlabel('x (deg)')
        ax_off.set_ylabel('y (deg)')
        #plt.colorbar(off_scatter, ax=ax_off, label='normalised spike count')
        def _update(frame):
            cx, cy = self.walk[frame]
            opt_img.set_extent([cx - half_x, cx + half_x,
                                 cy - half_y, cy + half_y])
            walk_line_opt.set_data(self.walk[:frame + 1, 0],
                                   self.walk[:frame + 1, 1])
            fovea_dot.set_data([cx], [cy])
            time_text.set_text(f't = {frame * self.dt:.3f} s')
            if glm_mode:
                on_now  = self.spikes_on[frame].astype(float)  
                off_now = self.spikes_off[frame].astype(float)
                on_norm  = on_now  / (on_now.max()  + 1e-9)
                off_norm = off_now / (off_now.max() + 1e-9)
                on_scatter.set_array(on_norm.ravel())
                off_scatter.set_array(off_norm.ravel())
            else:
                n_cols  = len(xg)
                on_vals = np.zeros(num_cells)
                for row_i, col_i in self.activated_indices[frame]:
                    on_vals[col_i * n_cols + row_i] = 1.0
                off_vals = 1.0 - on_vals
                on_scatter.set_array(on_vals)
                off_scatter.set_array(off_vals)
            walk_line_on.set_data(self.walk[:frame + 1, 0],
                                  self.walk[:frame + 1, 1])
            walk_line_off.set_data(self.walk[:frame + 1, 0],
                                   self.walk[:frame + 1, 1])
        anim = animation.FuncAnimation(
            fig, _update,
            frames=self.n_steps + 1,
            interval=interval, blit=False,
        )
        plt.tight_layout()
        if save_path:
            writer = "pillow" if save_path.endswith(".gif") else "ffmpeg"
            anim.save(save_path, writer=writer, fps=1000 // interval, dpi=120)
        return anim
if __name__ == "__main__":
    size = 28
    optotype = torch.zeros(size, size)
    c, t = size // 2, 2
    optotype[c - t // 2:c + t // 2 + 1, :] = 1.0
    optotype[:, c - t // 2:c + t // 2 + 1] = 1.0
    ds = 0.3
    sim = NeuralEncoder(dx=0.3, dy=0.3, dt=0.02, ds=ds)
    sim.fit(optotype, blur_sigma=1.0)
    sim.simulate_random_walk(sigma=0.1, T=1.0)
    sim.compute_activations(grid_range=10.0, grid_resolution=40, type='GLM')
    np.set_printoptions(threshold=np.inf)

    anim = sim.animate(interval=100)
    plt.show()
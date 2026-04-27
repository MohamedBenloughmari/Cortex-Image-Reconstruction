import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.datasets import fetch_openml


# ---------------- DICTIONARY LEARNING ---------------- #
def learn_mnist_dictionary(n_components=64, n_samples=5000,
                           alpha=1.0, max_iter=500, random_state=0):
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist.data[:n_samples].astype(np.float64) / 255.0
    mean_offset = float(X.mean())
    X -= X.mean(axis=1, keepdims=True)
    dico = MiniBatchDictionaryLearning(
        n_components=n_components, alpha=alpha, max_iter=max_iter,
        batch_size=64, random_state=random_state, transform_algorithm='lasso_cd'
    )
    dico.fit(X)
    D = dico.components_.T  # (784, n_components)
    D /= (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)
    return D, mean_offset


# ---------------- ENCODER + DECODER ---------------- #
class NeuralEncoder:
    def __init__(self, dx, dy, dt, D_diff, ds=None):
        self.dx, self.dy, self.dt = dx, dy, dt
        self.D_diff = D_diff
        self.ds = ds
        self.half_n = None
        self.optotype_np = None        # encoder frame: fliplr(img.T)
        self.optotype_display = None   # display frame
        self.n_steps = None
        self.walk = None
        self.ganglion_x = None
        self.ganglion_y = None
        self.spikes_on = None
        self.spikes_off = None

        # decoder state / history
        self.A_hat_history = None
        self.S_hat_history = None      # stored in MNIST orientation for display
        self.q_particles = None
        self.q_weights = None
        self._mean_offset = 0.0

        # GLM constants
        self.lambda0 = 10.0
        self.lambda1 = 100.0

    # ---------- coordinate-frame helpers ---------- #
    @staticmethod
    def _mnist_to_encoder(img):
        """(28,28) MNIST -> encoder frame used by GLM."""
        return np.fliplr(img.T)

    @staticmethod
    def _encoder_to_mnist(img):
        """inverse of _mnist_to_encoder."""
        return np.fliplr(img).T

    def _A_to_S_encoder(self, D, A):
        img = (D @ A).reshape(28, 28) + self._mean_offset
        return self._mnist_to_encoder(img)

    # ---------- ENCODER ---------- #
    def fit(self, optotype, blur_sigma=1.5):
        if optotype.dim() == 3:
            optotype = optotype.squeeze(0)
        h, _ = optotype.shape
        self.half_n = h // 2
        raw = optotype.numpy()
        blurred = gaussian_filter(raw, sigma=blur_sigma) if blur_sigma > 0 else raw
        self.optotype_display = blurred                       # MNIST frame (28,28)
        self.optotype_np = self._mnist_to_encoder(blurred)    # encoder frame

    def simulate_random_walk(self, T):
        self.n_steps = int(T / self.dt)
        sigma = np.sqrt(self.D_diff * self.dt)
        disp = np.random.normal(0.0, sigma, size=(self.n_steps, 2))
        self.walk = np.vstack([np.zeros((1, 2)), np.cumsum(disp, axis=0)])

    def _glm_rates(self, S, cx, cy):
        """S in encoder frame -> (lam_on, lam_off) on ganglion grid."""
        H, W = S.shape
        sigma_s = 0.5 * self.ds
        sigma_e = 0.203 * self.ds
        sigma2 = sigma_s**2 + sigma_e**2
        px_x = (np.arange(H) + 0.5 - self.half_n) * self.dx
        px_y = (np.arange(W) + 0.5 - self.half_n) * self.dy
        xg, yg = self.ganglion_x, self.ganglion_y
        diff_x = (cx + px_x[None, :]) - xg[:, None]
        diff_y = (cy + px_y[None, :]) - yg[:, None]
        gx_w = np.exp(-0.5 * diff_x**2 / sigma2)
        gy_w = np.exp(-0.5 * diff_y**2 / sigma2)
        c_raw = (gx_w @ S @ gy_w.T) / (2.0 * np.pi * sigma2)
        g_norm = max(c_raw.max(), 1e-9)
        c = np.clip(c_raw / g_norm, 0.0, 1.0)
        lam_on = self.lambda0 * np.exp(np.log(self.lambda1 / self.lambda0) * c)
        lam_off = self.lambda0 * np.exp(np.log(self.lambda1 / self.lambda0) * (1.0 - c))
        return lam_on, lam_off, gx_w, gy_w, g_norm, sigma2

    def compute_activations(self, grid_range=10.0, grid_resolution=40):
        self.grid_range = grid_range
        self.grid_resolution = grid_resolution
        self.ganglion_x = np.linspace(-grid_range, grid_range, grid_resolution)
        self.ganglion_y = np.linspace(-grid_range, grid_range, grid_resolution)
        n_t = self.n_steps + 1
        n_g = grid_resolution
        self.spikes_on = np.zeros((n_t, n_g, n_g), dtype=int)
        self.spikes_off = np.zeros((n_t, n_g, n_g), dtype=int)
        for t in range(n_t):
            cx, cy = self.walk[t]
            lam_on, lam_off, *_ = self._glm_rates(self.optotype_np, cx, cy)
            self.spikes_on[t] = np.random.poisson(lam_on * self.dt)
            self.spikes_off[t] = np.random.poisson(lam_off * self.dt)

    # ---------- DECODER ---------- #
    def _log_likelihood(self, lam_on, lam_off, r_on, r_off):
        return np.sum(r_on * np.log(lam_on * self.dt + 1e-12) - lam_on * self.dt
                    + r_off * np.log(lam_off * self.dt + 1e-12) - lam_off * self.dt)

    def _propagate_particles(self, particles, A_hat, D, t):
        n_p = particles.shape[0]
        sigma = np.sqrt(self.D_diff * self.dt)
        new_p = particles + np.random.normal(0.0, sigma, size=(n_p, 2))
        S = self._A_to_S_encoder(D, A_hat)
        r_on, r_off = self.spikes_on[t], self.spikes_off[t]
        log_w = np.zeros(n_p)
        for i in range(n_p):
            lam_on, lam_off, *_ = self._glm_rates(S, new_p[i, 0], new_p[i, 1])
            log_w[i] = self._log_likelihood(lam_on, lam_off, r_on, r_off)
        log_w -= log_w.max()
        w = np.exp(log_w); w /= w.sum()
        return new_p, w

    def _resample(self, particles, weights):
        n_p = len(weights)
        idx = np.searchsorted(np.cumsum(weights), np.random.uniform(0, 1, n_p))
        idx = np.clip(idx, 0, n_p - 1)
        return particles[idx], np.ones(n_p) / n_p

    def _sample_positions(self, particles, weights, n_samples):
        idx = np.searchsorted(np.cumsum(weights), np.random.uniform(0, 1, n_samples))
        idx = np.clip(idx, 0, len(weights) - 1)
        return particles[idx]

    def _grad_Er(self, A, D, samples_t, t):
        """Gradient of E_r w.r.t. A averaged over sampled positions."""
        S = self._A_to_S_encoder(D, A)
        r_on, r_off = self.spikes_on[t], self.spikes_off[t]
        log_ratio = np.log(self.lambda1 / self.lambda0)
        grad_S_enc = np.zeros_like(S)
        for cx, cy in samples_t:
            lam_on, lam_off, gx_w, gy_w, g_norm, sigma2 = self._glm_rates(S, cx, cy)
            dL_dc = (-(r_on - lam_on * self.dt) + (r_off - lam_off * self.dt)) * log_ratio
            grad_S_enc += (gx_w.T @ dL_dc @ gy_w) / (2.0 * np.pi * sigma2 * g_norm)
        grad_S_enc /= len(samples_t)
        # encoder frame -> MNIST frame -> sparse code
        grad_img = self._encoder_to_mnist(grad_S_enc)
        return D.T @ grad_img.ravel()

    @staticmethod
    def _soft_threshold(x, thresh):
        return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)

    def _fista_update(self, A_hat, H_hat, D, samples_t, t,
                      beta=0.05, gamma=0.1, n_iter=15, lr=0.01):
        A = A_hat.copy()
        Y = A.copy()
        t_k = 1.0
        for _ in range(n_iter):
            S_enc = self._A_to_S_encoder(D, Y)
            g_Eg = H_hat @ (Y - A_hat)
            g_Er = self._grad_Er(Y, D, samples_t, t)
            pen_enc = gamma * ((S_enc > 1).astype(float) - (S_enc < 0).astype(float))
            pen_img = self._encoder_to_mnist(pen_enc)
            g_pix = D.T @ pen_img.ravel()
            grad = g_Eg + g_Er + g_pix
            A_new = self._soft_threshold(Y - lr * grad, lr * beta)
            t_k_new = 0.5 * (1 + np.sqrt(1 + 4 * t_k**2))
            Y = A_new + ((t_k - 1) / t_k_new) * (A_new - A)
            A, t_k = A_new, t_k_new
        return A

    def _update_hessian(self, H_hat, D, tau=0.05, eps=1e-3):
        log_ratio = np.log(self.lambda1 / self.lambda0)
        scale = self.dt * 0.5 * (self.lambda0 + self.lambda1) * log_ratio**2
        JTJ = scale * (D.T @ D)
        return np.exp(-self.dt / tau) * H_hat + JTJ + eps * np.eye(D.shape[1])

    def decode(self, D, mean_offset=0.0, n_particles=80, n_samples=30,
               beta=0.05, gamma=0.1, fista_iter=10, lr=0.01, verbose=True):
        self._mean_offset = mean_offset
        N_sp = D.shape[1]
        A_hat = np.zeros(N_sp)
        H_hat = np.eye(N_sp) * 1e-3

        T = self.n_steps + 1
        particles = np.zeros((n_particles, 2))
        weights = np.ones(n_particles) / n_particles

        self.q_particles = np.zeros((T, n_particles, 2))
        self.q_weights = np.zeros((T, n_particles))
        self.A_hat_history = np.zeros((T, N_sp))
        self.S_hat_history = np.zeros((T, 28, 28))   # MNIST orientation

        for t in range(T):
            if t > 0:
                particles, weights = self._propagate_particles(particles, A_hat, D, t)
                particles, weights = self._resample(particles, weights)
            self.q_particles[t] = particles
            self.q_weights[t] = weights

            samples_t = self._sample_positions(particles, weights, n_samples)
            A_hat = self._fista_update(A_hat, H_hat, D, samples_t, t,
                                       beta=beta, gamma=gamma,
                                       n_iter=fista_iter, lr=lr)
            H_hat = self._update_hessian(H_hat, D)

            self.A_hat_history[t] = A_hat
            self.S_hat_history[t] = (D @ A_hat).reshape(28, 28) + self._mean_offset
            if verbose and t % max(T // 10, 1) == 0:
                print(f"[decode] t={t}/{T-1}  ||A||_1={np.abs(A_hat).sum():.2f}")
        return self.A_hat_history, self.S_hat_history

    # ---------- ANIMATION ---------- #
    def animate(self, interval=80, save_path=None):
        xg, yg = self.ganglion_x, self.ganglion_y
        half_x = self.half_n * self.dx
        half_y = self.half_n * self.dy
        X, Y = np.meshgrid(xg, yg, indexing='ij')
        pts_x, pts_y = X.ravel(), Y.ravel()
        n_cells = len(pts_x)

        has_decoder = self.S_hat_history is not None
        n_panels = 4 if has_decoder else 3
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
        ax_opt, ax_on, ax_off = axes[0], axes[1], axes[2]
        ax_rec = axes[3] if has_decoder else None
        for ax in axes:
            ax.set_aspect('equal')

        # --- stimulus + eye path ---
        cx0, cy0 = self.walk[0]
        opt_img = ax_opt.imshow(
            self.optotype_display,
            extent=[cx0 - half_x, cx0 + half_x, cy0 - half_y, cy0 + half_y],
            origin='upper', cmap='gray_r', alpha=0.9,
        )
        walk_line, = ax_opt.plot([], [], color='steelblue', lw=0.7)
        fovea, = ax_opt.plot([], [], 'b+', ms=12, mew=2)
        ax_opt.set_xlim(xg[0], xg[-1]); ax_opt.set_ylim(yg[0], yg[-1])
        ax_opt.set_title('Stimulus + eye path')
        time_text = ax_opt.text(0.02, 0.96, '', transform=ax_opt.transAxes, va='top')

        # --- ON / OFF spikes ---
        on_sc = ax_on.scatter(pts_x, pts_y, c=np.zeros(n_cells),
                              cmap='YlOrRd', vmin=0, vmax=1, s=18)
        ax_on.set_xlim(xg[0], xg[-1]); ax_on.set_ylim(yg[0], yg[-1])
        ax_on.set_title('ON spikes')

        off_sc = ax_off.scatter(pts_x, pts_y, c=np.zeros(n_cells),
                                cmap='YlGnBu', vmin=0, vmax=1, s=18)
        ax_off.set_xlim(xg[0], xg[-1]); ax_off.set_ylim(yg[0], yg[-1])
        ax_off.set_title('OFF spikes')

        # --- reconstruction ---
        if has_decoder:
            S0 = self.S_hat_history[0]
            vmax0 = max(abs(S0).max(), 1e-3)
            rec_img = ax_rec.imshow(S0, cmap='gray_r',
                                    vmin=0, vmax=vmax0, origin='upper')
            ax_rec.set_title(r'Reconstruction $D\hat{A}$')
            ax_rec.set_xticks([]); ax_rec.set_yticks([])

        def _update(frame):
            cx, cy = self.walk[frame]
            opt_img.set_extent([cx - half_x, cx + half_x, cy - half_y, cy + half_y])
            walk_line.set_data(self.walk[:frame + 1, 0], self.walk[:frame + 1, 1])
            fovea.set_data([cx], [cy])
            time_text.set_text(f't = {frame * self.dt:.3f} s')

            on_now = self.spikes_on[frame].astype(float)
            off_now = self.spikes_off[frame].astype(float)
            on_sc.set_array((on_now / (on_now.max() + 1e-9)).ravel())
            off_sc.set_array((off_now / (off_now.max() + 1e-9)).ravel())

            if has_decoder:
                S_hat = self.S_hat_history[frame]
                rec_img.set_data(S_hat)
                vmin, vmax = float(S_hat.min()), float(S_hat.max())
                if vmax - vmin < 1e-6:
                    vmax = vmin + 1e-3
                rec_img.set_clim(vmin, vmax)

            return opt_img, walk_line, fovea, on_sc, off_sc

        anim = animation.FuncAnimation(fig, _update,
                                       frames=self.n_steps + 1,
                                       interval=interval, blit=False)
        plt.tight_layout()
        if save_path:
            writer = "pillow" if save_path.endswith(".gif") else "ffmpeg"
            anim.save(save_path, writer=writer, fps=1000 // interval, dpi=120)
        return anim


# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    from torchvision import datasets, transforms

    print("Learning MNIST dictionary...")
    D, mean_offset = learn_mnist_dictionary(n_components=256, n_samples=60000,
                                            alpha=1.0, max_iter=300)
    print(f"D shape = {D.shape}, mean_offset = {mean_offset:.3f}")

    ds_val = datasets.MNIST(root='./data', train=False, download=True,
                            transform=transforms.ToTensor())
    optotype = ds_val[17][0].squeeze(0)

    sim = NeuralEncoder(dx=0.5, dy=0.5, dt=0.01, ds=0.3, D_diff=5.0)
    sim.fit(optotype, blur_sigma=0.0)
    sim.simulate_random_walk(T=1)
    sim.compute_activations(grid_range=10.0, grid_resolution=50)

    sim.decode(D, mean_offset=mean_offset,
               n_particles=80, n_samples=30,
               beta=1, gamma=0.1, fista_iter=50, lr=1e-6)

    anim = sim.animate(interval=100)
    plt.show()
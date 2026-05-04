import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.datasets import fetch_openml


def learn_mnist_dictionary(n_components=64, n_samples=5000,
                           alpha=1.0, max_iter=500, random_state=0):
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist.data[:n_samples].astype(np.float64) / 255.0
    mean_offset = float(X.mean())
    X = X - mean_offset
    dico = MiniBatchDictionaryLearning(
        n_components=n_components, alpha=alpha, max_iter=max_iter,
        batch_size=64, random_state=random_state, transform_algorithm='lasso_cd'
    )
    dico.fit(X)
    D = dico.components_.T
    D /= (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)
    return D, mean_offset


class NeuralEncoder:
    def __init__(self, dx, dy, dt, D_diff, ds=None, device=None):
        self.dx, self.dy, self.dt = dx, dy, dt
        self.D_diff = D_diff
        self.ds = ds
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.half_n = None
        self.optotype_np = None
        self.optotype_display = None
        self.n_steps = None
        self.walk = None
        self.ganglion_x = None
        self.ganglion_y = None
        self.spikes_on = None
        self.spikes_off = None
        self.r_on_all = None
        self.r_off_all = None

        self.A_hat_history = None
        self.S_hat_history = None
        self.q_particles = None
        self.q_weights = None
        self._mean_offset = 0.0

        self.lambda0 = 10.0
        self.lambda1 = 100.0
    @staticmethod
    def _mnist_to_encoder_np(img):
        return np.fliplr(img.T)
    @staticmethod
    def _mnist_to_encoder_torch(img):
        return torch.flip(img.T, dims=[-1])
    def fit(self, optotype, blur_sigma=1.5):
        if optotype.dim() == 3:
            optotype = optotype.squeeze(0)
        h, _ = optotype.shape
        self.half_n = h // 2
        raw = optotype.numpy()
        blurred = gaussian_filter(raw, sigma=blur_sigma) if blur_sigma > 0 else raw
        self.optotype_display = blurred
        self.optotype_np = self._mnist_to_encoder_np(blurred)

    def simulate_random_walk(self, T):
        self.n_steps = int(T / self.dt)
        sigma = np.sqrt(self.D_diff * self.dt)
        disp = np.random.normal(0.0, sigma, size=(self.n_steps, 2))
        self.walk = np.vstack([np.zeros((1, 2)), np.cumsum(disp, axis=0)])
    def _glm_rates_np(self, S, cx, cy):
        H, W = S.shape
        sigma_s = 0.5 * self.ds
        sigma_e = 0.203 * self.ds
        sigma2 = sigma_s ** 2 + sigma_e ** 2
        px_x = (np.arange(H) + 0.5 - self.half_n) * self.dx
        px_y = (np.arange(W) + 0.5 - self.half_n) * self.dy
        xg, yg = self.ganglion_x, self.ganglion_y
        diff_x = (cx + px_x[None, :]) - xg[:, None]
        diff_y = (cy + px_y[None, :]) - yg[:, None]
        gx_w = np.exp(-0.5 * diff_x ** 2 / sigma2)
        gy_w = np.exp(-0.5 * diff_y ** 2 / sigma2)
        c_raw = (gx_w @ S @ gy_w.T) / (2.0 * np.pi * sigma2)
        g_norm = max(c_raw.max(), 1e-9)
        c = np.clip(c_raw / g_norm, 0.0, 1.0)
        lam_on = self.lambda0 * np.exp(np.log(self.lambda1 / self.lambda0) * c)
        lam_off = self.lambda0 * np.exp(np.log(self.lambda1 / self.lambda0) * (1.0 - c))
        return lam_on, lam_off

    def _precompute_torch_constants(self):
        dev = self.device
        self.t_xg = torch.as_tensor(self.ganglion_x, dtype=torch.float32, device=dev)
        self.t_yg = torch.as_tensor(self.ganglion_y, dtype=torch.float32, device=dev)
        H = W = 2 * self.half_n
        self.t_px_x = (torch.arange(H, device=dev, dtype=torch.float32) + 0.5 - self.half_n) * self.dx
        self.t_px_y = (torch.arange(W, device=dev, dtype=torch.float32) + 0.5 - self.half_n) * self.dy
        sigma_s = 0.5 * self.ds
        sigma_e = 0.203 * self.ds
        self.t_sigma2 = torch.tensor(sigma_s ** 2 + sigma_e ** 2,
                                     dtype=torch.float32, device=dev)
        self.t_log_ratio = torch.tensor(np.log(self.lambda1 / self.lambda0),
                                        dtype=torch.float32, device=dev)
        self.t_two_pi_sigma2 = 2.0 * float(np.pi) * self.t_sigma2

    def _glm_rates_torch(self, S, cx, cy):
        """Original single-position version, kept for the Hessian path."""
        sigma2 = self.t_sigma2
        diff_x = (cx + self.t_px_x.unsqueeze(0)) - self.t_xg.unsqueeze(1)
        diff_y = (cy + self.t_px_y.unsqueeze(0)) - self.t_yg.unsqueeze(1)
        gx_w = torch.exp(-0.5 * diff_x ** 2 / sigma2)
        gy_w = torch.exp(-0.5 * diff_y ** 2 / sigma2)
        c_raw = (gx_w @ S @ gy_w.T) / (2.0 * np.pi * sigma2)
        g_norm = c_raw.detach().abs().max().clamp(min=1e-9)
        c = (c_raw / g_norm).clamp(0.0, 1.0)
        lam_on = self.lambda0 * torch.exp(self.t_log_ratio * c)
        lam_off = self.lambda0 * torch.exp(self.t_log_ratio * (1.0 - c))
        return lam_on, lam_off

    def _glm_rates_torch_batch(self, S, cx, cy):
        """Batched over B positions. Returns (B, n_g, n_g)."""
        sigma2 = self.t_sigma2
        diff_x = (cx[:, None, None] + self.t_px_x[None, None, :]) - self.t_xg[None, :, None]
        diff_y = (cy[:, None, None] + self.t_px_y[None, None, :]) - self.t_yg[None, :, None]
        gx_w = torch.exp(-0.5 * diff_x ** 2 / sigma2)   
        gy_w = torch.exp(-0.5 * diff_y ** 2 / sigma2)   
        tmp = torch.matmul(gx_w, S)                     
        c_raw = torch.einsum('bgw,bkw->bgk', tmp, gy_w) / self.t_two_pi_sigma2
        
        g_norm = c_raw.detach().amax(dim=(1, 2), keepdim=True).clamp(min=1e-9)
        c = (c_raw / g_norm).clamp(0.0, 1.0)
        lam_on = self.lambda0 * torch.exp(self.t_log_ratio * c)
        lam_off = self.lambda0 * torch.exp(self.t_log_ratio * (1.0 - c))
        return lam_on, lam_off

    def _A_to_S_encoder_torch(self, D_t, A):
        img = (D_t @ A).reshape(28, 28) + self._mean_offset
        return self._mnist_to_encoder_torch(img)

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
            lam_on, lam_off = self._glm_rates_np(self.optotype_np, cx, cy)
            self.spikes_on[t] = np.random.poisson(lam_on * self.dt)
            self.spikes_off[t] = np.random.poisson(lam_off * self.dt)

    def _propagate_particles(self, particles, A_hat_np, D, t):
        n_p = particles.shape[0]
        sigma = np.sqrt(self.D_diff * self.dt)
        new_p = particles + np.random.normal(0.0, sigma, size=(n_p, 2))
        S = np.fliplr(((D @ A_hat_np).reshape(28, 28) + self._mean_offset).T).copy()
        with torch.no_grad():
            S_t = torch.as_tensor(S, dtype=torch.float32, device=self.device)
            cx_t = torch.as_tensor(new_p[:, 0], dtype=torch.float32, device=self.device)
            cy_t = torch.as_tensor(new_p[:, 1], dtype=torch.float32, device=self.device)
            lam_on, lam_off = self._glm_rates_torch_batch(S_t, cx_t, cy_t)
            r_on = self.r_on_all[t]
            r_off = self.r_off_all[t]
            log_w = (r_on * torch.log(lam_on * self.dt + 1e-12) - lam_on * self.dt
                     + r_off * torch.log(lam_off * self.dt + 1e-12) - lam_off * self.dt
                     ).sum(dim=(1, 2))
            log_w = log_w - log_w.max()
            w = torch.exp(log_w)
            w = (w / w.sum()).cpu().numpy()
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

    def Er_fn(self, S, samples_t, r_on_t, r_off_t):
        cx = torch.as_tensor(samples_t[:, 0], dtype=torch.float32, device=self.device)
        cy = torch.as_tensor(samples_t[:, 1], dtype=torch.float32, device=self.device)
        lam_on, lam_off = self._glm_rates_torch_batch(S, cx, cy)
        dt = self.dt
        per_sample = (
            (lam_on * dt - r_on_t * torch.log(lam_on * dt + 1e-12)).sum(dim=(1, 2))
            + (lam_off * dt - r_off_t * torch.log(lam_off * dt + 1e-12)).sum(dim=(1, 2))
        )
        return per_sample.sum() / samples_t.shape[0]

    def Ep_fn(self, A_param, A_anchor, beta):
        log_p_anchor = -beta * torch.sum(torch.abs(A_anchor))
        grad_log_p = -beta * torch.sign(A_anchor)
        neg_Ep_lin = log_p_anchor + (grad_log_p * (A_param - A_anchor)).sum()
        return -neg_Ep_lin
    def _adam_update_A(self, A_param, Hessian, optimizer, D_t,
                       samples_t, r_on_t, r_off_t,
                       A_anchor, beta, gamma, n_iter, anchor_weight=1.0):
        last_loss = 0.0
        for _ in range(n_iter):
            optimizer.zero_grad()
            S = self._A_to_S_encoder_torch(D_t, A_param)
            Er = self.Er_fn(S, samples_t, r_on_t, r_off_t)

            diff = A_param - A_anchor
            Eg = 0.5 * anchor_weight * (diff @ (Hessian @ diff))
            Ep = self.Ep_fn(A_param, A_anchor, beta)

            range_pen = gamma * (
                torch.clamp(S - 1.0, min=0.0) ** 2
                + torch.clamp(-S, min=0.0) ** 2
            ).sum()

            loss = Er + Eg + Ep + range_pen
            loss.backward()
            optimizer.step()
            last_loss = loss.item()
        return last_loss
    @staticmethod
    def _hessian_of(scalar_fn, A_anchor):
        A = A_anchor.detach().clone().requires_grad_(True)
        val = scalar_fn(A)
        grad = torch.autograd.grad(val, A, create_graph=True)[0].flatten()
        n = A.numel()
        H = torch.zeros(n, n, device=A.device, dtype=A.dtype)
        for i in range(n):
            H[i] = torch.autograd.grad(grad[i], A, retain_graph=(i < n - 1))[0].flatten()
        return 0.5 * (H + H.T)

    def _update_hessian(self, H_prev, A_anchor, D_t, samples_t,
                        r_on_t, r_off_t, tau):
        def Er_of_A(A):
            S = self._A_to_S_encoder_torch(D_t, A)
            return self.Er_fn(S, samples_t, r_on_t, r_off_t)

        H_new = self._hessian_of(Er_of_A, A_anchor)
        decay = float(np.exp(-self.dt / tau))
        return decay * H_prev + H_new
    def decode(self, D, mean_offset=0.0, n_particles=80, n_samples=30,
               beta=0.05, gamma=0.1, adam_iter=20, lr=1e-2,
               anchor_weight=1.0, hessian_tau=0.5,
               hessian_every=1, verbose=True):
        self._mean_offset = mean_offset
        self._precompute_torch_constants()
        dev = self.device
        D_t = torch.as_tensor(D, dtype=torch.float32, device=dev)
        N_sp = D.shape[1]

        # Cache spikes on device once
        self.r_on_all = torch.as_tensor(self.spikes_on, dtype=torch.float32, device=dev)
        self.r_off_all = torch.as_tensor(self.spikes_off, dtype=torch.float32, device=dev)

        A_param = nn.Parameter(torch.zeros(N_sp, device=dev, dtype=torch.float32))
        Hessian = torch.zeros((N_sp, N_sp), device=dev, dtype=torch.float32)
        optimizer = torch.optim.Adam([A_param], lr=lr)

        T = self.n_steps + 1
        particles = np.zeros((n_particles, 2))
        weights = np.ones(n_particles) / n_particles

        self.q_particles = np.zeros((T, n_particles, 2))
        self.q_weights = np.zeros((T, n_particles))
        self.A_hat_history = np.zeros((T, N_sp))
        self.S_hat_history = np.zeros((T, 28, 28))

        for t in range(T):
            cur_anchor_w = 0.0 if t == 0 else anchor_weight

            if t > 0:
                A_np = A_param.detach().cpu().numpy()
                particles, weights = self._propagate_particles(particles, A_np, D, t)
                particles, weights = self._resample(particles, weights)
            self.q_particles[t] = particles
            self.q_weights[t] = weights

            samples_t = self._sample_positions(particles, weights, n_samples)
            r_on_t = self.r_on_all[t]
            r_off_t = self.r_off_all[t]
            A_anchor = A_param.detach().clone()

            loss_val = self._adam_update_A(
                A_param, Hessian, optimizer, D_t, samples_t, r_on_t, r_off_t,
                A_anchor=A_anchor, anchor_weight=cur_anchor_w,
                beta=beta, gamma=gamma, n_iter=adam_iter,
            )

            if (t % hessian_every) == 0:
                A_new = A_param.detach().clone()
                Hessian = self._update_hessian(
                    Hessian, A_new, D_t, samples_t, r_on_t, r_off_t,
                    tau=hessian_tau,
                )

            A_np = A_param.detach().cpu().numpy()
            self.A_hat_history[t] = A_np
            self.S_hat_history[t] = (D @ A_np).reshape(28, 28) + self._mean_offset

            if verbose and t % max(T // 10, 1) == 0:
                print(f"[decode] t={t}/{T-1}  loss={loss_val:.3f}  "
                      f"||A||_1={np.abs(A_np).sum():.2f}")
        return self.A_hat_history, self.S_hat_history

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

        on_sc = ax_on.scatter(pts_x, pts_y, c=np.zeros(n_cells),
                              cmap='YlOrRd', vmin=0, vmax=1, s=18)
        ax_on.set_xlim(xg[0], xg[-1]); ax_on.set_ylim(yg[0], yg[-1])
        ax_on.set_title('ON spikes')

        off_sc = ax_off.scatter(pts_x, pts_y, c=np.zeros(n_cells),
                                cmap='YlGnBu', vmin=0, vmax=1, s=18)
        ax_off.set_xlim(xg[0], xg[-1]); ax_off.set_ylim(yg[0], yg[-1])
        ax_off.set_title('OFF spikes')

        on_global = max(self.spikes_on.max(), 1)
        off_global = max(self.spikes_off.max(), 1)

        if has_decoder:
            S0 = self.S_hat_history[0]
            vmax0 = max(abs(S0).max(), 1e-3)
            rec_img = ax_rec.imshow(S0, cmap='gray_r', vmin=0, vmax=vmax0, origin='upper')
            ax_rec.set_title(r'Reconstruction $D\hat{A}$')
            ax_rec.set_xticks([]); ax_rec.set_yticks([])

        def _update(frame):
            cx, cy = self.walk[frame]
            opt_img.set_extent([cx - half_x, cx + half_x, cy - half_y, cy + half_y])
            walk_line.set_data(self.walk[:frame + 1, 0], self.walk[:frame + 1, 1])
            fovea.set_data([cx], [cy])
            time_text.set_text(f't = {frame * self.dt:.3f} s')
            on_sc.set_array((self.spikes_on[frame].astype(float) / on_global).ravel())
            off_sc.set_array((self.spikes_off[frame].astype(float) / off_global).ravel())
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


if __name__ == "__main__":
    from torchvision import datasets, transforms

    print("Learning MNIST dictionary")
    D, mean_offset = learn_mnist_dictionary(n_components=256, n_samples=60000,
                                            alpha=1.0, max_iter=300)
    print(f"D shape = {D.shape}, mean_offset = {mean_offset:.3f}")

    ds_val = datasets.MNIST(root='./data', train=False, download=True,
                            transform=transforms.ToTensor())
    optotype = ds_val[1][0].squeeze(0)

    sim = NeuralEncoder(dx=0.3, dy=0.3, dt=0.01, ds=0.3, D_diff=0.0)
    sim.fit(optotype, blur_sigma=0.0)
    sim.simulate_random_walk(T=1)
    sim.compute_activations(grid_range=10.0, grid_resolution=20)
    sim.decode(
        D, mean_offset=mean_offset,
        n_particles=100, n_samples=100,
        beta=0.001, gamma=0.01,
        adam_iter=50, lr=1e-3,
        anchor_weight=1.0,
        hessian_tau=15,
        hessian_every=1,
    )
    anim = sim.animate(interval=100,save_path='d0.gif')
    plt.show()
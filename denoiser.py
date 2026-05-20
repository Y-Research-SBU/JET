import torch
import torch.nn as nn
import torch.nn.functional as F

from models.raw_vit import RawViTDiffusion


class Denoiser(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.net = RawViTDiffusion(
            model_name=args.model,
            num_channels=args.num_eeg_channels,
            patch_size=args.eeg_patch_size,
            target_length=args.target_length,
            num_classes=args.class_num,
            attn_dropout=args.attn_dropout,
            proj_dropout=args.proj_dropout,
        )
        self.sample_shape = (args.num_eeg_channels, args.eeg_patch_num, args.eeg_patch_size)
        self.raw_vit_num_channels = args.num_eeg_channels
        self.raw_vit_target_length = args.target_length
        self.raw_vit_patch_size = args.eeg_patch_size
        self.raw_vit_patch_num = args.eeg_patch_num

        self.num_classes = args.class_num

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale
        self.noise_type = getattr(args, "noise_type", "gs")
        self.loss_type = getattr(args, "loss_type", "l2")
        self.loss_weight_stft = getattr(args, "loss_weight_stft", 0.2)
        self.loss_weight_stat = getattr(args, "loss_weight_stat", 1.0)
        self.loss_weight_tv = getattr(args, "loss_weight_tv", 0.05)
        self.loss_weight_corr = getattr(args, "loss_weight_corr", 0.05)

        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        return torch.where(drop, torch.full_like(labels, self.num_classes), labels)

    def sample_t(self, n: int, device=None, noise_type: str = None):
        noise_type = noise_type or self.noise_type
        if noise_type == "zero":
            return torch.ones(n, device=device)
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def to_training_space(self, batch: torch.Tensor) -> torch.Tensor:
        return self._raw_vit_to_patches(batch)

    def _raw_vit_to_patches(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.dim() == 4 and batch.shape[1] == self.raw_vit_num_channels and batch.shape[2] == self.raw_vit_patch_num and batch.shape[3] == self.raw_vit_patch_size:
            return batch
        if batch.dim() == 3 and batch.shape[0] == self.raw_vit_num_channels and batch.shape[1] == self.raw_vit_patch_num and batch.shape[2] == self.raw_vit_patch_size:
            return batch

        added_batch_dim = False
        if batch.dim() == 2:
            batch = batch.unsqueeze(0)
            added_batch_dim = True
        elif batch.dim() == 3 and batch.shape[0] == self.raw_vit_num_channels:
            batch = batch.unsqueeze(0)
            added_batch_dim = True

        if batch.dim() < 3:
            raise ValueError("Expected raw EEG tensors with at least 3 dims (batch, channels, length).")
        if batch.shape[1] != self.raw_vit_num_channels:
            raise ValueError(f"Expected {self.raw_vit_num_channels} EEG channels but received {batch.shape[1]}.")

        total = batch.shape[-1]
        target = self.raw_vit_target_length
        if total < target:
            batch = F.pad(batch, (0, target - total))
        elif total > target:
            batch = batch[..., :target]

        batch = batch.view(batch.shape[0], self.raw_vit_num_channels, self.raw_vit_patch_num, self.raw_vit_patch_size)

        if added_batch_dim:
            batch = batch.squeeze(0)
        return batch

    def forward(self, x, labels):
        x = self.to_training_space(x)
        labels_dropped = self.drop_labels(labels) if self.training else labels

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = self._noise_like(x)

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        x_pred = self.net(z, t.flatten(), labels_dropped)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        return self._compute_loss(v, v_pred, x, x_pred)

    def _compute_loss(self, v, v_pred, x, x_pred):
        loss_l1 = (x - x_pred).abs().mean()

        if self.loss_type == "mix":
            mix_loss = loss_l1

            if self.loss_weight_stft > 0:
                fft_sizes = [64, 128, 256, 512, 1024]
                hop_sizes = [16, 32, 64, 128, 256]
                win_lengths = [64, 128, 256, 512, 1024]

                loss_stft = 0.0
                for n_fft, hop, win in zip(fft_sizes, hop_sizes, win_lengths):
                    loss_stft += self._stft_loss_multires(x, x_pred, n_fft, hop, win)
                loss_stft /= len(fft_sizes)
                mix_loss = mix_loss + self.loss_weight_stft * loss_stft

            if self.loss_weight_stat > 0:
                mix_loss = mix_loss + self.loss_weight_stat * self._statistics_loss(x, x_pred)

            if self.loss_weight_tv > 0:
                mix_loss = mix_loss + self.loss_weight_tv * self._total_variation_loss(x_pred)

            if self.loss_weight_corr > 0:
                mix_loss = mix_loss + self.loss_weight_corr * self._pearson_corr_loss(x, x_pred)

            return mix_loss

        if self.loss_type == "l1":
            return (v - v_pred).abs().mean()
        return ((v - v_pred) ** 2).mean()

    @staticmethod
    def _stft_loss_multires(x, x_pred, n_fft, hop_length, win_length):
        window = torch.hann_window(win_length).to(x.device)

        def get_magnitude(tensor):
            b, c = tensor.shape[:2]
            tensor = tensor.view(b * c, -1).float()
            stft = torch.stft(tensor, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)
            return torch.abs(stft)

        mag_x = get_magnitude(x)
        mag_pred = get_magnitude(x_pred)

        sc_loss = torch.norm(mag_x - mag_pred, p="fro") / (torch.norm(mag_x, p="fro") + 1e-7)
        lm_loss = F.l1_loss(torch.log(mag_x + 1e-7), torch.log(mag_pred + 1e-7))
        return sc_loss + lm_loss

    @staticmethod
    def _statistics_loss(x, x_pred):
        std_x = x.std(dim=-1)
        std_pred = x_pred.std(dim=-1)
        mean_x = x.mean(dim=-1)
        mean_pred = x_pred.mean(dim=-1)
        return F.l1_loss(std_pred, std_x) + F.l1_loss(mean_pred, mean_x)

    @staticmethod
    def _total_variation_loss(x):
        return (x[..., 1:] - x[..., :-1]).abs().mean()

    @staticmethod
    def _pearson_corr_loss(x, x_pred):
        x_f = x.reshape(x.shape[0], -1)
        pred_f = x_pred.reshape(x_pred.shape[0], -1)
        x_center = x_f - x_f.mean(dim=1, keepdim=True)
        pred_center = pred_f - pred_f.mean(dim=1, keepdim=True)
        numerator = (x_center * pred_center).sum(dim=1)
        denominator = torch.sqrt((x_center.pow(2).sum(dim=1) + 1e-8) * (pred_center.pow(2).sum(dim=1) + 1e-8))
        corr = numerator / denominator
        return (1.0 - corr.clamp(-1.0, 1.0)).mean()

    def _noise_like(self, x):
        if self.noise_type == "zero":
            return torch.zeros_like(x)
        return torch.randn_like(x) * self.noise_scale

    @torch.no_grad()
    def update_ema(self):
        for p, p_ema1, p_ema2 in zip(self.parameters(), self.ema_params1, self.ema_params2):
            p_ema1.mul_(self.ema_decay1).add_(p.data, alpha=1.0 - self.ema_decay1)
            p_ema2.mul_(self.ema_decay2).add_(p.data, alpha=1.0 - self.ema_decay2)

    @torch.no_grad()
    def _run_net(self, z, t, labels, use_ema=True):
        if not use_ema:
            return self.net(z, t, labels)

        backup = [p.detach().clone() for p in self.parameters()]
        ema = self.ema_params1 if self.ema_params1 is not None else backup
        for p, p_ema in zip(self.parameters(), ema):
            p.data.copy_(p_ema)

        out = self.net(z, t, labels)

        for p, old in zip(self.parameters(), backup):
            p.data.copy_(old)
        return out

    @torch.no_grad()
    def generate(self, labels, cfg=None, noise_type=None):
        b = labels.shape[0]
        device = labels.device
        cfg_scale = self.cfg_scale if cfg is None else cfg

        z = self._noise_like(torch.empty((b,) + self.sample_shape, device=device))
        if noise_type == "zero":
            z = torch.zeros_like(z)

        steps = self.steps
        use_heun = self.method == "heun"
        t_schedule = torch.linspace(0.0, 1.0, steps + 1, device=device)

        def _get_x_pred(z_in, t_scalar):
            t_vec = torch.full((b,), t_scalar, device=device)
            x_cond = self._run_net(z_in, t_vec, labels, use_ema=True)
            if cfg_scale > 1.0:
                null_labels = torch.full_like(labels, self.num_classes)
                x_uncond = self._run_net(z_in, t_vec, null_labels, use_ema=True)
                return x_uncond + cfg_scale * (x_cond - x_uncond)
            return x_cond

        for i in range(steps):
            t_cur = float(t_schedule[i])
            t_nxt = float(t_schedule[i + 1])
            dt = t_nxt - t_cur

            x1 = _get_x_pred(z, t_cur)
            v1 = (x1 - z) / max(self.t_eps, (1.0 - t_cur))

            if use_heun and i < steps - 1:
                z_euler = z + dt * v1
                x2 = _get_x_pred(z_euler, t_nxt)
                v2 = (x2 - z_euler) / max(self.t_eps, (1.0 - t_nxt))
                z = z + dt * 0.5 * (v1 + v2)
            else:
                z = z + dt * v1

        return z

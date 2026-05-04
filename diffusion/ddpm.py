from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from diffusion.schedule import DiffusionSchedule


@dataclass
class DDPMOutput:
    loss: torch.Tensor
    eps_pred: torch.Tensor
    eps_true: torch.Tensor
    x_t: torch.Tensor
    t: torch.Tensor


class DDPM:
    def __init__(self, schedule: DiffusionSchedule):
        self.s = schedule

    @staticmethod
    def scale_to_neg_one_to_one(x01: torch.Tensor) -> torch.Tensor:
        # [0,1] -> [-1,1]
        return x01 * 2.0 - 1.0

    @staticmethod
    def unscale_to_zero_one(x11: torch.Tensor) -> torch.Tensor:
        # [-1,1] -> [0,1]
        return (x11 + 1.0) / 2.0

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_ab = self.s.extract(self.s.sqrt_alpha_bars, t, x0.shape)
        sqrt_1mab = self.s.extract(self.s.sqrt_one_minus_alpha_bars, t, x0.shape)
        return sqrt_ab * x0 + sqrt_1mab * noise

    def p_losses(
        self,
        model,
        x0: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        lambda_bce: float = 0.05,
        lambda_dice: float = 0.05,
    ) -> DDPMOutput:
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0=x0, t=t, noise=noise)
        eps_pred = model(x_t=x_t, cond=cond, t=t)

        mse_map = (eps_pred - noise) ** 2  # (B,1,H,W)

        if weight is None:
            loss = mse_map.mean()
        else:
            loss = (mse_map * weight).sum() / torch.clamp(weight.sum(), min=1.0)
        x0_pred = self.predict_x0_from_eps(x_t=x_t, t=t, eps=eps_pred)
        x0_pred_01 = self.unscale_to_zero_one(x0_pred).clamp(1e-4, 1 - 1e-4)
        gt01 = self.unscale_to_zero_one(x0).clamp(0.0, 1.0)

        with torch.amp.autocast("cuda", enabled=False):
            bce_map = F.binary_cross_entropy(
                x0_pred_01.float(),
                gt01.float(),
                reduction="none"
            ) # (B,1,H,W)

        if weight is None:
            bce = bce_map.mean()
            p = x0_pred_01
            g = gt01
        else:
            bce = (bce_map * weight).sum() / torch.clamp(weight.sum(), min=1.0)
            p = x0_pred_01 * weight
            g = gt01 * weight

        eps_ = 1e-7
        inter = (p * g).sum(dim=(1, 2, 3))
        den = (p + g).sum(dim=(1, 2, 3))
        dice_loss = (1.0 - (2 * inter + eps_) / (den + eps_)).mean()

        loss = loss + lambda_bce * bce + lambda_dice * dice_loss

        return DDPMOutput(loss=loss, eps_pred=eps_pred, eps_true=noise, x_t=x_t, t=t)

    @torch.no_grad()
    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self.s.extract(self.s.sqrt_alpha_bars, t, x_t.shape)
        sqrt_1mab = self.s.extract(self.s.sqrt_one_minus_alpha_bars, t, x_t.shape)
        x0 = (x_t - sqrt_1mab * eps) / torch.clamp(sqrt_ab, min=1e-8)
        return x0

    @torch.no_grad()
    def p_sample_ddpm(self, model, x_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor, clip_denoised: bool = True):

        eps = model(x_t=x_t, cond=cond, t=t)
        x0_pred = self.predict_x0_from_eps(x_t=x_t, t=t, eps=eps)
        if clip_denoised:
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        coef1 = self.s.extract(self.s.posterior_mean_coef1, t, x_t.shape)
        coef2 = self.s.extract(self.s.posterior_mean_coef2, t, x_t.shape)
        mean = coef1 * x0_pred + coef2 * x_t

        log_var = self.s.extract(self.s.posterior_log_var, t, x_t.shape)

        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float()
        while nonzero_mask.dim() < x_t.dim():
            nonzero_mask = nonzero_mask.unsqueeze(-1)

        x_prev = mean + nonzero_mask * torch.exp(0.5 * log_var) * noise
        return x_prev, x0_pred

    @torch.no_grad()
    def sample_ddpm(
        self,
        model,
        cond: torch.Tensor,
        shape: Tuple[int, int, int, int],
        steps: Optional[int] = None,
        clip_denoised: bool = True
    ) -> torch.Tensor:
        device = cond.device
        T = self.s.timesteps if steps is None else steps

        x_t = torch.randn(shape, device=device)
        for i in reversed(range(T)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x_t, x0_pred = self.p_sample_ddpm(model, x_t, cond, t, clip_denoised=clip_denoised)

        return x0_pred

    @torch.no_grad()
    def sample_ddim(
        self,
        model,
        cond: torch.Tensor,
        shape: Tuple[int, int, int, int],
        steps: int = 50,
        eta: float = 0.0,
        clip_denoised: bool = True
    ) -> torch.Tensor:
        device = cond.device
        T = self.s.timesteps

        times = torch.linspace(0, T - 1, steps, device=device).long()
        times = times.flip(0)

        x = torch.randn(shape, device=device)

        for idx, t_scalar in enumerate(times):
            t = torch.full((shape[0],), int(t_scalar.item()), device=device, dtype=torch.long)
            eps = model(x_t=x, cond=cond, t=t)
            x0_pred = self.predict_x0_from_eps(x_t=x, t=t, eps=eps)
            if clip_denoised:
                x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

            if idx == len(times) - 1:
                x = x0_pred
                break

            t_next = torch.full((shape[0],), int(times[idx + 1].item()), device=device, dtype=torch.long)

            ab_t = self.s.extract(self.s.alpha_bars, t, x.shape)
            ab_next = self.s.extract(self.s.alpha_bars, t_next, x.shape)

            sigma = eta * torch.sqrt((1 - ab_next) / (1 - ab_t) * (1 - ab_t / ab_next))
            noise = torch.randn_like(x)

            x = torch.sqrt(ab_next) * x0_pred + torch.sqrt(1 - ab_next - sigma ** 2) * eps + sigma * noise

        return x0_pred
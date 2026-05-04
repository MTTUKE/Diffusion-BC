import math
import torch


def linear_beta_schedule(
    timesteps: int,
    beta_start: float,
    beta_end: float,
    device: torch.device
) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32, device=device)


def cosine_beta_schedule(
    timesteps: int,
    s: float,
    device: torch.device
) -> torch.Tensor:
    """
    Cosine schedule from Nichol & Dhariwal.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32, device=device)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clamp(betas, min=1e-6, max=0.999)
    return betas


class DiffusionSchedule:
    def __init__(
        self,
        timesteps: int,
        beta_start: float,
        beta_end: float,
        device: torch.device,
        schedule_type: str = "linear",
        cosine_s: float = 0.008
    ):
        self.timesteps = timesteps
        self.device = device
        self.schedule_type = schedule_type

        if schedule_type == "cosine":
            betas = cosine_beta_schedule(timesteps, cosine_s, device)
        elif schedule_type == "linear":
            betas = linear_beta_schedule(timesteps, beta_start, beta_end, device)
        else:
            raise ValueError(f"Unknown schedule_type: {schedule_type}")

        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars

        self.sqrt_alphas = torch.sqrt(alphas)
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

        alpha_bars_prev = torch.cat([torch.tensor([1.0], device=device), alpha_bars[:-1]], dim=0)
        self.posterior_var = betas * (1.0 - alpha_bars_prev) / torch.clamp((1.0 - alpha_bars), min=1e-20)
        self.posterior_log_var = torch.log(torch.clamp(self.posterior_var, min=1e-20))

        self.posterior_mean_coef1 = betas * torch.sqrt(alpha_bars_prev) / torch.clamp((1.0 - alpha_bars), min=1e-20)
        self.posterior_mean_coef2 = (1.0 - alpha_bars_prev) * torch.sqrt(alphas) / torch.clamp((1.0 - alpha_bars), min=1e-20)

    def extract(self, a: torch.Tensor, t: torch.Tensor, x_shape):
        out = a.gather(0, t).float()
        while out.dim() < len(x_shape):
            out = out.unsqueeze(-1)
        return out
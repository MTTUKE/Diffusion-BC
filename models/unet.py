import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, device=t.device).float() / (half - 1)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.time_proj = nn.Linear(time_dim, out_ch)

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class ConditionalUNet(nn.Module):
    def __init__(self, base_ch: int = 64, dropout: float = 0.1, time_dim: int = 256):
        super().__init__()
        self.time_dim = time_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        in_ch = 2
        out_ch = 1

        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        self.down1 = ResBlock(base_ch, base_ch, time_dim, dropout)
        self.down2 = ResBlock(base_ch, base_ch * 2, time_dim, dropout)
        self.down3 = ResBlock(base_ch * 2, base_ch * 4, time_dim, dropout)
        self.down4 = ResBlock(base_ch * 4, base_ch * 8, time_dim, dropout)

        self.ds1 = Downsample(base_ch)
        self.ds2 = Downsample(base_ch * 2)
        self.ds3 = Downsample(base_ch * 4)

        self.mid1 = ResBlock(base_ch * 8, base_ch * 8, time_dim, dropout)
        self.mid2 = ResBlock(base_ch * 8, base_ch * 8, time_dim, dropout)


        self.us1 = Upsample(base_ch * 8)  # 32 - 64
        self.us2 = Upsample(base_ch * 4)  # 64 - 128
        self.us3 = Upsample(base_ch * 2)  # 128 - 256

        self.up1 = ResBlock(base_ch * 8 + base_ch * 4, base_ch * 4, time_dim, dropout)

        self.up2 = ResBlock(base_ch * 4 + base_ch * 2, base_ch * 2, time_dim, dropout)

        self.up3 = ResBlock(base_ch * 2 + base_ch, base_ch, time_dim, dropout)

        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, out_ch, 3, padding=1)

    def forward(self, x_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        x = torch.cat([x_t, cond], dim=1)
        x0 = self.in_conv(x)

        d1 = self.down1(x0, t_emb)
        x1 = self.ds1(d1)

        d2 = self.down2(x1, t_emb)
        x2 = self.ds2(d2)

        d3 = self.down3(x2, t_emb)
        x3 = self.ds3(d3)

        d4 = self.down4(x3, t_emb)

        m = self.mid1(d4, t_emb)
        m = self.mid2(m, t_emb)

        u1 = self.us1(m)
        u1 = torch.cat([u1, d3], dim=1)  # skip 64x64
        u1 = self.up1(u1, t_emb)

        u2 = self.us2(u1)
        u2 = torch.cat([u2, d2], dim=1)  # skip 128x128
        u2 = self.up2(u2, t_emb)

        u3 = self.us3(u2)
        u3 = torch.cat([u3, d1], dim=1)  # skip 256x256
        u3 = self.up3(u3, t_emb)

        out = self.out_conv(F.silu(self.out_norm(u3)))
        return out

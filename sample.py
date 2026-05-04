import os
import argparse
from typing import List

import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from data.dataset import SCSSFolderDataset
from diffusion.schedule import DiffusionSchedule
from diffusion.ddpm import DDPM
from models.unet import ConditionalUNet
from utils.metrics import dice_iou
from utils.vis import save_triplet


def apply_tta_image(img: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return img
    if mode == "hflip":
        return torch.flip(img, dims=[-1])
    if mode == "vflip":
        return torch.flip(img, dims=[-2])
    if mode == "hvflip":
        return torch.flip(img, dims=[-2, -1])
    raise ValueError(f"Unknown TTA mode: {mode}")


def invert_tta_mask(mask: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return mask
    if mode == "hflip":
        return torch.flip(mask, dims=[-1])
    if mode == "vflip":
        return torch.flip(mask, dims=[-2])
    if mode == "hvflip":
        return torch.flip(mask, dims=[-2, -1])
    raise ValueError(f"Unknown TTA mode: {mode}")


def get_tta_modes(tta: str) -> List[str]:
    if tta == "none":
        return ["none"]
    if tta == "flip4":
        return ["none", "hflip", "vflip", "hvflip"]
    raise ValueError(f"Unknown tta mode: {tta}")


def postprocess_mask(
    prob01: np.ndarray,
    threshold: float,
    kernel_size: int,
    min_area: int
) -> np.ndarray:
    import cv2

    mask = (prob01 > threshold).astype(np.uint8)

    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if min_area > 0:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = np.zeros_like(mask)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                out[labels == i] = 1
        mask = out

    return mask.astype(np.float32)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoints/best.pt or epoch_*.pt")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--ddim", action="store_true")
    parser.add_argument("--ddim_steps", type=int, default=100)
    parser.add_argument("--thr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--tta", type=str, default="none", choices=["none", "flip4"])
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--min_area", type=int, default=200)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(args.ckpt, map_location=device)
    train_cfg = ckpt.get("train_config", {})

    timesteps = int(train_cfg.get("timesteps", cfg.timesteps))
    beta_start = float(train_cfg.get("beta_start", cfg.beta_start))
    beta_end = float(train_cfg.get("beta_end", cfg.beta_end))
    schedule_type = str(train_cfg.get("schedule_type", getattr(cfg, "schedule_type", "linear")))
    cosine_s = float(train_cfg.get("cosine_s", getattr(cfg, "cosine_s", 0.008)))
    base_channels = int(train_cfg.get("base_channels", cfg.base_channels))
    dropout = float(train_cfg.get("dropout", cfg.dropout))
    disk_thr = float(train_cfg.get("disk_thr", 0.05))

    print(
        f"[CKPT cfg] timesteps={timesteps} beta_start={beta_start} "
        f"beta_end={beta_end} schedule={schedule_type} cosine_s={cosine_s} "
        f"base_ch={base_channels} dropout={dropout} disk_thr={disk_thr}"
    )

    test_imgs = os.path.join(cfg.data_root, cfg.test_imgs_dir)
    test_masks = os.path.join(cfg.data_root, cfg.test_masks_dir)
    test_ds = SCSSFolderDataset(test_imgs, test_masks, img_size=cfg.img_size, augmentations=None)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    model = ConditionalUNet(base_ch=base_channels, dropout=dropout).to(device)

    if "ema" in ckpt and ckpt["ema"] is not None:
        model.load_state_dict(ckpt["ema"])
        print("Loaded EMA weights for sampling.")
    else:
        model.load_state_dict(ckpt["model"])
        print("Loaded raw model weights for sampling.")

    model.eval()

    best_thr = ckpt.get("best_thr", 0.5)
    if args.thr is not None:
        best_thr = float(args.thr)
    print(f"Using threshold: {best_thr:.2f}")
    print(f"Using TTA: {args.tta}")
    print(f"Using kernel_size: {args.kernel_size}")
    print(f"Using min_area: {args.min_area}")

    schedule = DiffusionSchedule(
        timesteps=timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        device=device,
        schedule_type=schedule_type,
        cosine_s=cosine_s
    )
    ddpm = DDPM(schedule)

    os.makedirs(args.out_dir, exist_ok=True)

    dices: List[float] = []
    ious: List[float] = []

    tta_modes = get_tta_modes(args.tta)

    for idx, sample in enumerate(tqdm(test_loader, desc="Sampling")):
        img = sample["image"].to(device)
        gt = sample["mask"].to(device)
        name = sample["name"][0]

        disk = (img > disk_thr).float()

        preds01 = []
        for tta_idx, mode in enumerate(tta_modes):
            img_tta = apply_tta_image(img, mode)
            cond = ddpm.scale_to_neg_one_to_one(img_tta)

            for sample_idx in range(args.n_samples):
                current_seed = args.seed + idx * 10000 + tta_idx * 1000 + sample_idx
                torch.manual_seed(current_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(current_seed)
                random.seed(current_seed)
                np.random.seed(current_seed)

                if args.ddim:
                    x0_pred = ddpm.sample_ddim(
                        model=model,
                        cond=cond,
                        shape=gt.shape,
                        steps=args.ddim_steps,
                        eta=0.0
                    )
                else:
                    x0_pred = ddpm.sample_ddpm(
                        model=model,
                        cond=cond,
                        shape=gt.shape
                    )

                pred01 = ddpm.unscale_to_zero_one(x0_pred).clamp(0.0, 1.0)
                pred01 = invert_tta_mask(pred01, mode)
                preds01.append(pred01)

        preds01_t = torch.stack(preds01, dim=0)
        mean01 = preds01_t.mean(dim=0)
        std01 = preds01_t.std(dim=0)

        mean_np = mean01[0, 0].detach().cpu().numpy()
        pred_np = postprocess_mask(
            prob01=mean_np,
            threshold=best_thr,
            kernel_size=args.kernel_size,
            min_area=args.min_area
        )

        pred_bin = torch.from_numpy(pred_np)[None, None].to(device)

        d, i = dice_iou(pred_bin * disk, gt * disk)
        dices.append(d)
        ious.append(i)

        img_np = img[0, 0].detach().cpu().numpy()
        gt_np = gt[0, 0].detach().cpu().numpy()
        unc_np = std01[0, 0].detach().cpu().numpy()

        out_path = os.path.join(args.out_dir, f"{name}.png")
        save_triplet(out_path, img_np, gt_np, pred_np, unc01=unc_np)

    print(f"Test Dice (mean over samples): {sum(dices) / len(dices):.4f}")
    print(f"Test IoU  (mean over samples): {sum(ious) / len(ious):.4f}")


if __name__ == "__main__":
    main()
import os
import argparse
import random
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config as OldConfig
from config_clean import Config as CleanConfig

from utils.seed import seed_everything
from utils.metrics import dice_iou
from data.dataset import (
    SCSSFolderDataset,
    build_default_augmentations,
    ManifestSegmentationDataset,
    build_clean_augmentations,
)
from diffusion.schedule import DiffusionSchedule
from diffusion.ddpm import DDPM
from models.unet import ConditionalUNet
from utils.ema import EMA


DISK_THR = 0.05
T_BIAS_PROB = 0.45


def make_loaders(cfg, pin_memory: bool, mode: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if mode == "old":
        train_imgs = os.path.join(cfg.data_root, cfg.train_imgs_dir)
        train_masks = os.path.join(cfg.data_root, cfg.train_masks_dir)

        test_imgs = os.path.join(cfg.data_root, cfg.test_imgs_dir)
        test_masks = os.path.join(cfg.data_root, cfg.test_masks_dir)

        aug = build_default_augmentations()

        base_dataset = SCSSFolderDataset(
            train_imgs,
            train_masks,
            img_size=cfg.img_size,
            augmentations=None
        )

        n_total = len(base_dataset)
        n_val = int(n_total * cfg.val_fraction)
        n_train = n_total - n_val

        generator = torch.Generator().manual_seed(cfg.split_seed)
        perm = torch.randperm(n_total, generator=generator).tolist()

        train_indices = perm[:n_train]
        val_indices = perm[n_train:]

        train_dataset_full = SCSSFolderDataset(
            train_imgs,
            train_masks,
            img_size=cfg.img_size,
            augmentations=aug
        )
        val_dataset_full = SCSSFolderDataset(
            train_imgs,
            train_masks,
            img_size=cfg.img_size,
            augmentations=None
        )

        train_ds = Subset(train_dataset_full, train_indices)
        val_ds = Subset(val_dataset_full, val_indices)

        test_ds = SCSSFolderDataset(
            test_imgs,
            test_masks,
            img_size=cfg.img_size,
            augmentations=None
        )

    else:
        manifest_csv = os.path.join(cfg.data_root, cfg.manifest_csv)

        train_ds = ManifestSegmentationDataset(
            manifest_csv=manifest_csv,
            split="train",
            img_size=cfg.img_size,
            augmentations=build_clean_augmentations(),
            source_filter=cfg.source_filter
        )

        val_ds = ManifestSegmentationDataset(
            manifest_csv=manifest_csv,
            split="val",
            img_size=cfg.img_size,
            augmentations=None,
            source_filter=cfg.source_filter
        )

        test_ds = ManifestSegmentationDataset(
            manifest_csv=manifest_csv,
            split="test",
            img_size=cfg.img_size,
            augmentations=None,
            source_filter=cfg.source_filter
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader


@torch.no_grad()
def quick_val_metrics(model, ddpm: DDPM, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    dice_sum, iou_sum, n = 0.0, 0.0, 0

    for batch in loader:
        img = batch["image"].to(device)
        mask = batch["mask"].to(device)

        if "disk_mask" in batch:
            disk = batch["disk_mask"].to(device)
        else:
            disk = (img > DISK_THR).float()

        cond = ddpm.scale_to_neg_one_to_one(img)
        x0 = ddpm.scale_to_neg_one_to_one(mask)

        B = x0.shape[0]
        t = torch.randint(0, ddpm.s.timesteps, (B,), device=device, dtype=torch.long)

        out = ddpm.p_losses(model, x0=x0, cond=cond, t=t, weight=disk)

        x0_pred = ddpm.predict_x0_from_eps(out.x_t, t, out.eps_pred)
        x0_pred_01 = ddpm.unscale_to_zero_one(x0_pred)
        pred_bin = (x0_pred_01 > 0.5).float()

        d, i = dice_iou(pred_bin * disk, mask * disk)
        dice_sum += d
        iou_sum += i
        n += 1

    return dice_sum / max(n, 1), iou_sum / max(n, 1)


@torch.no_grad()
def sampling_val_metrics(
    model_for_sampling,
    ddpm: DDPM,
    val_sample_loader: DataLoader,
    device: torch.device,
    ddim_steps: int,
    n_samples: int,
    thr_grid: List[float],
    max_images: int
) -> Tuple[float, float, float]:
    model_for_sampling.eval()

    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    py_state = random.getstate()
    np_state = np.random.get_state()

    sum_dice_per_thr = {thr: 0.0 for thr in thr_grid}
    sum_iou_per_thr = {thr: 0.0 for thr in thr_grid}
    count = 0

    for batch in val_sample_loader:
        seed = 1234 + count
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)

        img = batch["image"].to(device)
        gt = batch["mask"].to(device)

        if "disk_mask" in batch:
            disk = batch["disk_mask"].to(device)
        else:
            disk = (img > DISK_THR).float()

        cond = ddpm.scale_to_neg_one_to_one(img)

        preds = []
        for _ in range(n_samples):
            x0_pred = ddpm.sample_ddim(
                model=model_for_sampling,
                cond=cond,
                shape=gt.shape,
                steps=ddim_steps,
                eta=0.0
            )
            pred01 = ddpm.unscale_to_zero_one(x0_pred).clamp(0.0, 1.0)
            preds.append(pred01)

        preds_t = torch.stack(preds, dim=0)
        mean01 = preds_t.mean(dim=0)

        for thr in thr_grid:
            pred_bin = (mean01 > thr).float()
            d, i = dice_iou(pred_bin * disk, gt * disk)
            sum_dice_per_thr[thr] += d
            sum_iou_per_thr[thr] += i

        count += 1
        if count >= max_images:
            break

    best_thr = max(thr_grid, key=lambda t: sum_dice_per_thr[t] / max(count, 1))
    best_dice = sum_dice_per_thr[best_thr] / max(count, 1)
    best_iou = sum_iou_per_thr[best_thr] / max(count, 1)

    expected_dice = (2 * best_iou) / (1 + best_iou + 1e-12)
    print(f"[SANITY] sampling: dice={best_dice:.4f}, iou={best_iou:.4f}, expected_dice_from_iou={expected_dice:.4f}")

    torch.random.set_rng_state(torch_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)
    random.setstate(py_state)
    np.random.set_state(np_state)

    return best_dice, best_iou, best_thr


def save_ckpt(
    path: str,
    model,
    optimizer,
    epoch: int,
    best_score: float,
    best_thr: float,
    ema_state=None,
    train_config: Optional[Dict[str, Any]] = None
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "ema": ema_state,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_score": best_score,
        "best_thr": best_thr,
        "train_config": train_config or {},
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["old", "clean"], default="clean")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--base_channels", type=int, default=None)
    parser.add_argument("--no_amp", action="store_true")
    args = parser.parse_args()

    cfg = OldConfig() if args.mode == "old" else CleanConfig()

    if args.data_root is not None:
        cfg.data_root = args.data_root
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.timesteps is not None:
        cfg.timesteps = args.timesteps
    if args.base_channels is not None:
        cfg.base_channels = args.base_channels
    if args.no_amp:
        cfg.use_amp = False

    seed_everything(cfg.split_seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")

    pin = (device.type == "cuda")
    train_loader, val_loader, test_loader = make_loaders(cfg, pin_memory=pin, mode=args.mode)

    val_sample_loader = DataLoader(
        val_loader.dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=pin
    )

    model = ConditionalUNet(base_ch=cfg.base_channels, dropout=cfg.dropout).to(device)
    ema = EMA(model, decay=cfg.ema_decay)

    schedule = DiffusionSchedule(
        timesteps=cfg.timesteps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        device=device,
        schedule_type=cfg.schedule_type,
        cosine_s=cfg.cosine_s
    )
    ddpm = DDPM(schedule)

    print(f"alpha_bar_T = {schedule.alpha_bars[-1].item():.12e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
        eta_min=cfg.lr * 0.1
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.use_amp and use_cuda))

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=cfg.log_dir)

    sample_every = cfg.sample_val_every_epochs
    sample_images = cfg.sample_val_images
    val_ddim_steps = cfg.val_ddim_steps
    val_n_samples = cfg.val_n_samples

    thr_grid = [round(x, 2) for x in torch.arange(
        cfg.thr_grid_start,
        cfg.thr_grid_end + 1e-9,
        cfg.thr_grid_step
    ).tolist()]

    train_config = {
        "mode": args.mode,
        "timesteps": int(cfg.timesteps),
        "beta_start": float(cfg.beta_start),
        "beta_end": float(cfg.beta_end),
        "schedule_type": str(cfg.schedule_type),
        "cosine_s": float(cfg.cosine_s),
        "base_channels": int(cfg.base_channels),
        "dropout": float(cfg.dropout),
        "img_size": int(cfg.img_size),
        "disk_thr": float(DISK_THR),
        "source_filter": getattr(cfg, "source_filter", "all"),
        "manifest_csv": getattr(cfg, "manifest_csv", ""),
    }

    best_sampling_dice = -1.0
    best_thr = 0.5
    no_improve_count = 0
    stop_training = False

    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}")

        for batch in pbar:
            img = batch["image"].to(device)
            mask = batch["mask"].to(device)

            if "disk_mask" in batch:
                disk = batch["disk_mask"].to(device)
            else:
                disk = (img > DISK_THR).float()

            cond = ddpm.scale_to_neg_one_to_one(img)
            x0 = ddpm.scale_to_neg_one_to_one(mask)

            B = x0.shape[0]

            if torch.rand(1).item() < T_BIAS_PROB:
                t = torch.randint(cfg.timesteps // 2, cfg.timesteps, (B,), device=device, dtype=torch.long)
            else:
                t = torch.randint(0, cfg.timesteps, (B,), device=device, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(cfg.use_amp and use_cuda)):
                out = ddpm.p_losses(model, x0=x0, cond=cond, t=t, weight=disk)
                loss = out.loss

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            ema.update(model)

            global_step += 1
            pbar.set_postfix(loss=float(loss.item()))
            writer.add_scalar("train/loss", loss.item(), global_step)

        val_dice, val_iou = quick_val_metrics(model, ddpm, val_loader, device)
        writer.add_scalar("val/quick_dice", val_dice, epoch)
        writer.add_scalar("val/quick_iou", val_iou, epoch)
        print(f"[Epoch {epoch}] quick_val Dice={val_dice:.4f} IoU={val_iou:.4f}")

        if sample_every > 0 and (epoch % sample_every == 0):
            s_dice, s_iou, s_thr = sampling_val_metrics(
                model_for_sampling=ema.ema_model,
                ddpm=ddpm,
                val_sample_loader=val_sample_loader,
                device=device,
                ddim_steps=val_ddim_steps,
                n_samples=val_n_samples,
                thr_grid=thr_grid,
                max_images=sample_images
            )

            writer.add_scalar("val/sampling_dice", s_dice, epoch)
            writer.add_scalar("val/sampling_iou", s_iou, epoch)
            writer.add_scalar("val/best_thr", s_thr, epoch)

            print(f"[Epoch {epoch}] SAMPLING val Dice={s_dice:.4f} IoU={s_iou:.4f} best_thr={s_thr:.2f}")

            if s_dice > best_sampling_dice + cfg.early_stop_min_delta:
                best_sampling_dice = s_dice
                best_thr = s_thr
                no_improve_count = 0

                save_ckpt(
                    os.path.join(cfg.ckpt_dir, "best.pt"),
                    model,
                    optimizer,
                    epoch,
                    best_sampling_dice,
                    best_thr,
                    ema_state=ema.state_dict(),
                    train_config=train_config
                )
            else:
                no_improve_count += 1
                print(f"[INFO] no improvement count: {no_improve_count}/{cfg.early_stop_patience}")

                if no_improve_count >= cfg.early_stop_patience:
                    print("[EARLY STOP] Sampling metric stopped improving.")
                    stop_training = True

        scheduler.step()
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        if epoch % cfg.save_every_epochs == 0:
            save_ckpt(
                os.path.join(cfg.ckpt_dir, f"epoch_{epoch}.pt"),
                model,
                optimizer,
                epoch,
                best_sampling_dice,
                best_thr,
                ema_state=ema.state_dict(),
                train_config=train_config
            )

        if stop_training:
            break

    test_dice, test_iou = quick_val_metrics(model, ddpm, test_loader, device)
    print(f"[FINAL] quick_test Dice={test_dice:.4f} IoU={test_iou:.4f}")

    writer.close()


if __name__ == "__main__":
    main()
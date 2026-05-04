from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

import torch

THIS_FILE = Path(__file__).resolve()
if THIS_FILE.parent.name == "eval_reiss":
    PROJECT_ROOT = THIS_FILE.parent.parent
else:
    PROJECT_ROOT = THIS_FILE.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Config  # noqa: E402
from diffusion.schedule import DiffusionSchedule  # noqa: E402
from diffusion.ddpm import DDPM  # noqa: E402
from models.unet import ConditionalUNet  # noqa: E402

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Diffusion-BC inference on Reiss benchmark.")

    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to manifest.csv created by prepare_manifest.py",
    )
    parser.add_argument(
        "--reiss-root",
        type=str,
        default=None,
        help="Prepared dataset root. If omitted, parent of manifest.csv is used.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to checkpoint, usually checkpoints/best.pt",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Directory where inference outputs will be saved.",
    )
    parser.add_argument(
        "--method-name",
        type=str,
        default="Diffusion-BC",
        help="Method name stored in metadata.",
    )

    parser.add_argument(
        "--n-samples",
        dest="n_samples",
        type=int,
        default=6,
        help="How many stochastic diffusion samples to average per TTA branch.",
    )
    parser.add_argument(
        "--ddim",
        action="store_true",
        help="Use DDIM sampling instead of full DDPM.",
    )
    parser.add_argument(
        "--ddim-steps",
        dest="ddim_steps",
        type=int,
        default=100,
        help="Number of DDIM steps when --ddim is enabled.",
    )
    parser.add_argument(
        "--thr",
        type=float,
        default=0.72,
        help="Probability threshold used for final binary mask.",
    )
    parser.add_argument(
        "--tta",
        type=str,
        default="flip4",
        choices=["none", "flip4"],
        help="Test-time augmentation mode.",
    )
    parser.add_argument(
        "--kernel-size",
        dest="kernel_size",
        type=int,
        default=5,
        help="Morphological closing kernel size.",
    )
    parser.add_argument(
        "--min-area",
        dest="min_area",
        type=int,
        default=200,
        help="Remove connected components smaller than this area.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed.",
    )

    parser.add_argument(
        "--source-disk-radius",
        dest="source_disk_radius",
        type=int,
        default=404,
        help="Solar disk radius in raw 1024x1024 Reiss images. "
             "404 matches your custom_data_preparation.ipynb.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Debug mode: process only first N cases.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into existing non-empty output directory.",
    )

    return parser.parse_args()

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def ensure_out_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {path}\n"
            f"Use --overwrite to continue."
        )
    path.mkdir(parents=True, exist_ok=True)


def load_manifest_rows(manifest_path: Path) -> List[Dict[str, str]]:
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Manifest is empty: {manifest_path}")
    return rows


def float01_to_uint8(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


def save_gray_png(path: Path, x01: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(float01_to_uint8(x01), mode="L").save(path)


def save_rgb_png(path: Path, x_rgb_u8: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(x_rgb_u8.astype(np.uint8), mode="RGB").save(path)


def save_binary_mask_png(path: Path, mask01: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask_u8 = ((mask01 > 0.5).astype(np.uint8) * 255)
    Image.fromarray(mask_u8, mode="L").save(path)


def resize_mask_to_shape(mask01: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    pil = Image.fromarray(((mask01 > 0.5).astype(np.uint8) * 255), mode="L")
    pil = pil.resize((w, h), Image.NEAREST)
    return (np.asarray(pil, dtype=np.float32) > 127.0).astype(np.float32)

def get_tta_modes(tta: str) -> List[str]:
    if tta == "none":
        return ["none"]
    if tta == "flip4":
        return ["none", "hflip", "vflip", "hvflip"]
    raise ValueError(f"Unknown TTA mode: {tta}")


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

def postprocess_mask(prob01: np.ndarray, threshold: float, kernel_size: int, min_area: int) -> np.ndarray:
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

def make_circular_disk_mask(size_hw: Tuple[int, int], radius_px: int) -> np.ndarray:
    h, w = size_hw
    cx, cy = w // 2, h // 2

    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse(
        (cx - radius_px, cy - radius_px, cx + radius_px, cy + radius_px),
        fill=255,
    )
    return (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)


def preprocess_reiss_like_scss(
    image_path: Path,
    img_size: int = 256,
    source_disk_radius_1024: int = 404,
) -> Dict[str, np.ndarray]:
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        rgb_np = np.asarray(rgb, dtype=np.uint8)
        h, w = rgb_np.shape[:2]

        # Масштабируем радиус, если картинка не 1024x1024.
        scale = min(h, w) / 1024.0
        radius_src = int(round(source_disk_radius_1024 * scale))

        # Маска диска в исходном размере.
        disk_mask_src = make_circular_disk_mask((h, w), radius_src)

        # Вне солнечного диска делаем белый фон,
        # как у твоих подготовленных SCSS картинок.
        white_bg = np.full_like(rgb_np, 255, dtype=np.uint8)
        disk_mask_src_3 = disk_mask_src[..., None]
        cropped_rgb_np = np.where(disk_mask_src_3 > 0.5, rgb_np, white_bg)

        cropped_rgb = Image.fromarray(cropped_rgb_np, mode="RGB")

        # Теперь идём точно по смыслу dataset.py:
        # convert("L") -> resize to 256 -> divide by 255.
        gray = cropped_rgb.convert("L").resize((img_size, img_size), Image.BILINEAR)
        gray_np = np.asarray(gray, dtype=np.float32) / 255.0

        # Делаем disk mask уже в размере модели.
        radius_model = int(round(radius_src * img_size / min(h, w)))
        disk_mask_model = make_circular_disk_mask((img_size, img_size), radius_model).astype(np.float32)

        # После resize на границе могут появиться серые полутоновые артефакты.
        # Снаружи диска снова жёстко ставим белый фон.
        gray_np = np.where(disk_mask_model > 0.5, gray_np, 1.0).astype(np.float32)

        return {
            "original_rgb_np": rgb_np,
            "cropped_rgb_np": cropped_rgb_np,
            "model_input_np": gray_np,
            "disk_mask_np": disk_mask_model,
        }
def load_model_and_ddpm(ckpt_path: Path, device: torch.device, cfg: Config):
    ckpt = torch.load(ckpt_path, map_location=device)
    train_cfg = ckpt.get("train_config", {})

    timesteps = int(train_cfg.get("timesteps", cfg.timesteps))
    beta_start = float(train_cfg.get("beta_start", cfg.beta_start))
    beta_end = float(train_cfg.get("beta_end", cfg.beta_end))
    schedule_type = str(train_cfg.get("schedule_type", getattr(cfg, "schedule_type", "linear")))
    cosine_s = float(train_cfg.get("cosine_s", getattr(cfg, "cosine_s", 0.008)))
    base_channels = int(train_cfg.get("base_channels", cfg.base_channels))
    dropout = float(train_cfg.get("dropout", cfg.dropout))
    disk_thr = float(train_cfg.get("disk_thr", 0.05))

    model = ConditionalUNet(base_ch=base_channels, dropout=dropout).to(device)

    if "ema" in ckpt and ckpt["ema"] is not None:
        model.load_state_dict(ckpt["ema"])
        weights_source = "ema"
    else:
        model.load_state_dict(ckpt["model"])
        weights_source = "raw"

    model.eval()

    schedule = DiffusionSchedule(
        timesteps=timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        device=device,
        schedule_type=schedule_type,
        cosine_s=cosine_s,
    )
    ddpm = DDPM(schedule)

    best_thr = ckpt.get("best_thr", None)

    runtime_cfg = {
        "timesteps": timesteps,
        "beta_start": beta_start,
        "beta_end": beta_end,
        "schedule_type": schedule_type,
        "cosine_s": cosine_s,
        "base_channels": base_channels,
        "dropout": dropout,
        "disk_thr": disk_thr,
        "weights_source": weights_source,
        "ckpt_best_thr": best_thr,
    }

    return ckpt, model, ddpm, runtime_cfg


@torch.no_grad()
def infer_single_probability_map(
    model,
    ddpm: DDPM,
    img_t: torch.Tensor,
    args: argparse.Namespace,
    case_seed_offset: int,
) -> Tuple[np.ndarray, np.ndarray]:
    tta_modes = get_tta_modes(args.tta)
    preds01 = []

    for tta_idx, mode in enumerate(tta_modes):
        img_tta = apply_tta_image(img_t, mode)
        cond = ddpm.scale_to_neg_one_to_one(img_tta)

        for sample_idx in range(args.n_samples):
            current_seed = args.seed + case_seed_offset * 10000 + tta_idx * 1000 + sample_idx
            set_seed(current_seed)

            if args.ddim:
                x0_pred = ddpm.sample_ddim(
                    model=model,
                    cond=cond,
                    shape=img_t.shape,
                    steps=args.ddim_steps,
                    eta=0.0,
                )
            else:
                x0_pred = ddpm.sample_ddpm(
                    model=model,
                    cond=cond,
                    shape=img_t.shape,
                )

            pred01 = ddpm.unscale_to_zero_one(x0_pred).clamp(0.0, 1.0)
            pred01 = invert_tta_mask(pred01, mode)
            preds01.append(pred01)

    preds01_t = torch.stack(preds01, dim=0)
    mean01 = preds01_t.mean(dim=0)
    std01 = preds01_t.std(dim=0)

    mean_np = mean01[0, 0].detach().cpu().numpy().astype(np.float32)
    std_np = std01[0, 0].detach().cpu().numpy().astype(np.float32)

    return mean_np, std_np
def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    reiss_root = Path(args.reiss_root).expanduser().resolve() if args.reiss_root else manifest_path.parent
    if not reiss_root.is_dir():
        raise FileNotFoundError(f"Reiss root not found: {reiss_root}")

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    ensure_out_dir(out_dir, overwrite=args.overwrite)

    cases_dir = out_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt, model, ddpm, runtime_cfg = load_model_and_ddpm(ckpt_path, device, cfg)

    print("[Checkpoint config]")
    for k, v in runtime_cfg.items():
        print(f"  {k}: {v}")

    print("[Inference config]")
    print(f"  method_name   : {args.method_name}")
    print(f"  n_samples     : {args.n_samples}")
    print(f"  ddim          : {args.ddim}")
    print(f"  ddim_steps    : {args.ddim_steps}")
    print(f"  thr           : {args.thr}")
    print(f"  tta           : {args.tta}")
    print(f"  kernel_size   : {args.kernel_size}")
    print(f"  min_area      : {args.min_area}")
    print(f"  disk_radius   : {args.source_disk_radius}")

    rows = load_manifest_rows(manifest_path)
    if args.limit is not None:
        rows = rows[:args.limit]
        print(f"Debug limit enabled: {len(rows)} case(s) will be processed.")

    prediction_rows: List[Dict[str, object]] = []

    for idx, row in enumerate(tqdm(rows, desc="Reiss inference")):
        case_id = row["case_id"]

        image_rel = row.get("image_relpath") or row.get("input_relpath")
        if not image_rel:
            raise KeyError(f"Manifest row for case {case_id} has neither image_relpath nor input_relpath")

        image_path = reiss_root / image_rel
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found for case {case_id}: {image_path}")
        prep = preprocess_reiss_like_scss(
            image_path=image_path,
            img_size=cfg.img_size,
            source_disk_radius_1024=args.source_disk_radius,
        )

        original_rgb_np = prep["original_rgb_np"]
        cropped_rgb_np = prep["cropped_rgb_np"]
        img01_np = prep["model_input_np"]
        disk_mask_np = prep["disk_mask_np"]

        orig_h, orig_w = original_rgb_np.shape[:2]

        img_t = torch.from_numpy(img01_np)[None, None, ...].to(device)

        mean_prob_np, std_prob_np = infer_single_probability_map(
            model=model,
            ddpm=ddpm,
            img_t=img_t,
            args=args,
            case_seed_offset=idx,
        )

        mean_prob_np = (mean_prob_np * disk_mask_np).astype(np.float32)
        std_prob_np = (std_prob_np * disk_mask_np).astype(np.float32)

        mask256_np = postprocess_mask(
            prob01=mean_prob_np,
            threshold=args.thr,
            kernel_size=args.kernel_size,
            min_area=args.min_area,
        )

        mask256_np = (mask256_np * disk_mask_np).astype(np.float32)
        mask512_np = resize_mask_to_shape(mask256_np, (orig_h, orig_w))

        case_dir = cases_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        save_rgb_png(case_dir / "input_original.png", original_rgb_np)
        save_rgb_png(case_dir / "cropped_input_original.png", cropped_rgb_np)
        save_gray_png(case_dir / "input_256.png", img01_np)
        save_binary_mask_png(case_dir / "disk_mask_256.png", disk_mask_np)

        np.save(case_dir / "prob_256.npy", mean_prob_np)
        save_gray_png(case_dir / "prob_256.png", mean_prob_np)

        np.save(case_dir / "uncertainty_256.npy", std_prob_np)
        save_gray_png(case_dir / "uncertainty_256.png", std_prob_np)

        save_binary_mask_png(case_dir / "mask_256.png", mask256_np)
        save_binary_mask_png(case_dir / "mask_512.png", mask512_np)

        pixel_count_256 = int(mask256_np.sum())
        pixel_count_512 = int(mask512_np.sum())

        inside = disk_mask_np > 0.5
        outside = disk_mask_np <= 0.5

        mean_prob_inside = float(mean_prob_np[inside].mean()) if inside.any() else 0.0
        mean_prob_outside = float(mean_prob_np[outside].mean()) if outside.any() else 0.0
        max_prob_inside = float(mean_prob_np[inside].max()) if inside.any() else 0.0
        max_prob_outside = float(mean_prob_np[outside].max()) if outside.any() else 0.0

        mean_uncertainty = float(std_prob_np[inside].mean()) if inside.any() else 0.0
        max_uncertainty = float(std_prob_np[inside].max()) if inside.any() else 0.0

        case_meta = {
            "case_id": case_id,
            "method_name": args.method_name,
            "image_path": str(image_path),
            "image_original_width": orig_w,
            "image_original_height": orig_h,
            "model_input_size": cfg.img_size,
            "preprocessing": {
                "description": "SCSS-like preprocessing for raw Reiss inputs",
                "source_disk_radius_1024": int(args.source_disk_radius),
                "outside_disk_fill": "white",
                "grayscale_after_crop": True,
                "resize_method": "bilinear",
                "disk_mask_applied_before_threshold": True,
                "disk_mask_applied_after_postprocess": True,
            },
            "sampling": {
                "ddim": bool(args.ddim),
                "ddim_steps": int(args.ddim_steps),
                "n_samples": int(args.n_samples),
                "tta": args.tta,
                "seed": int(args.seed),
            },
            "postprocess": {
                "threshold": float(args.thr),
                "kernel_size": int(args.kernel_size),
                "min_area": int(args.min_area),
            },
            "runtime_cfg": runtime_cfg,
            "stats": {
                "predicted_pixels_256": pixel_count_256,
                "predicted_pixels_512": pixel_count_512,
                "mean_probability_inside_disk_256": mean_prob_inside,
                "mean_probability_outside_disk_256": mean_prob_outside,
                "max_probability_inside_disk_256": max_prob_inside,
                "max_probability_outside_disk_256": max_prob_outside,
                "mean_uncertainty_inside_disk_256": mean_uncertainty,
                "max_uncertainty_inside_disk_256": max_uncertainty,
            },
            "files": {
                "input_original": "input_original.png",
                "cropped_input_original": "cropped_input_original.png",
                "input_256": "input_256.png",
                "disk_mask_256": "disk_mask_256.png",
                "prob_256_npy": "prob_256.npy",
                "prob_256_png": "prob_256.png",
                "uncertainty_256_npy": "uncertainty_256.npy",
                "uncertainty_256_png": "uncertainty_256.png",
                "mask_256": "mask_256.png",
                "mask_512": "mask_512.png",
            },
        }

        (case_dir / "meta.json").write_text(json.dumps(case_meta, indent=2), encoding="utf-8")

        prediction_rows.append({
            "case_id": case_id,
            "image_relpath": row.get("image_relpath") or row.get("input_relpath") or "",
            "input_relpath": row.get("input_relpath") or row.get("image_relpath") or "",
            "annotation_relpath": row.get("annotation_relpath", ""),
            "case_output_relpath": case_dir.relative_to(out_dir).as_posix(),
            "input_256_relpath": (case_dir / "input_256.png").relative_to(out_dir).as_posix(),
            "mask_256_relpath": (case_dir / "mask_256.png").relative_to(out_dir).as_posix(),
            "mask_512_relpath": (case_dir / "mask_512.png").relative_to(out_dir).as_posix(),
            "prob_256_npy_relpath": (case_dir / "prob_256.npy").relative_to(out_dir).as_posix(),
            "uncertainty_256_npy_relpath": (case_dir / "uncertainty_256.npy").relative_to(out_dir).as_posix(),
            "predicted_pixels_256": pixel_count_256,
            "predicted_pixels_512": pixel_count_512,
            "mean_probability_inside_disk_256": f"{mean_prob_inside:.6f}",
            "mean_probability_outside_disk_256": f"{mean_prob_outside:.6f}",
            "mean_uncertainty_inside_disk_256": f"{mean_uncertainty:.6f}",
            "max_uncertainty_inside_disk_256": f"{max_uncertainty:.6f}",
        })

    predictions_manifest_path = out_dir / "predictions_manifest.csv"
    fieldnames = [
        "case_id",
        "image_relpath",
        "input_relpath",
        "annotation_relpath",
        "case_output_relpath",
        "input_256_relpath",
        "mask_256_relpath",
        "mask_512_relpath",
        "prob_256_npy_relpath",
        "uncertainty_256_npy_relpath",
        "predicted_pixels_256",
        "predicted_pixels_512",
        "mean_probability_inside_disk_256",
        "mean_probability_outside_disk_256",
        "mean_uncertainty_inside_disk_256",
        "max_uncertainty_inside_disk_256",
    ]

    with predictions_manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prediction_rows)

    total_pred_pixels_256 = int(sum(int(r["predicted_pixels_256"]) for r in prediction_rows))
    total_pred_pixels_512 = int(sum(int(r["predicted_pixels_512"]) for r in prediction_rows))

    avg_mean_prob_inside = float(np.mean([float(r["mean_probability_inside_disk_256"]) for r in prediction_rows])) if prediction_rows else 0.0
    avg_mean_prob_outside = float(np.mean([float(r["mean_probability_outside_disk_256"]) for r in prediction_rows])) if prediction_rows else 0.0
    avg_mean_unc = float(np.mean([float(r["mean_uncertainty_inside_disk_256"]) for r in prediction_rows])) if prediction_rows else 0.0

    run_summary = {
        "method_name": args.method_name,
        "manifest_path": str(manifest_path),
        "reiss_root": str(reiss_root),
        "checkpoint_path": str(ckpt_path),
        "output_dir": str(out_dir),
        "device": str(device),
        "n_cases_processed": len(prediction_rows),
        "model_input_size": cfg.img_size,
        "source_disk_radius_1024": int(args.source_disk_radius),
        "sampling": {
            "ddim": bool(args.ddim),
            "ddim_steps": int(args.ddim_steps),
            "n_samples": int(args.n_samples),
            "tta": args.tta,
            "seed": int(args.seed),
        },
        "postprocess": {
            "threshold": float(args.thr),
            "kernel_size": int(args.kernel_size),
            "min_area": int(args.min_area),
        },
        "checkpoint_runtime_cfg": runtime_cfg,
        "aggregate_stats": {
            "total_predicted_pixels_256": total_pred_pixels_256,
            "total_predicted_pixels_512": total_pred_pixels_512,
            "average_mean_probability_inside_disk_256": avg_mean_prob_inside,
            "average_mean_probability_outside_disk_256": avg_mean_prob_outside,
            "average_mean_uncertainty_inside_disk_256": avg_mean_unc,
        },
        "notes": [
            "This version preprocesses raw Reiss inputs to match SCSS-like training data more closely.",
            "Outside the solar disk is filled with white before grayscale conversion and resizing.",
            "Probability and final binary mask are both explicitly clipped by the disk mask.",
            "This script prepares inference outputs only; object-level manual Reiss scoring is a separate step.",
        ],
    }

    (out_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"Predictions manifest: {predictions_manifest_path}")
    print(f"Run summary         : {out_dir / 'run_summary.json'}")
    print(f"Cases directory     : {cases_dir}")


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build visual review panels for comparison_29 Reiss benchmark.")
    p.add_argument("--manifest", required=True, help="manifest.csv from prepare_manifest.py")
    p.add_argument("--predictions-manifest", required=True, help="predictions_manifest.csv from infer_reiss.py")
    p.add_argument("--reiss-root", default=None, help="Prepared dataset root. Default: manifest parent")
    p.add_argument("--predictions-root", default=None, help="Inference root. Default: predictions manifest parent")
    p.add_argument("--out-dir", required=True, help="Output folder for review panels")
    p.add_argument("--method-name", default="Diffusion-BC")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"CSV is empty: {path}")
    return rows


def ensure_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {path}\nUse --overwrite to continue.")
    path.mkdir(parents=True, exist_ok=True)


def open_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def open_mask_rgb(path: Path) -> Image.Image:
    mask = Image.open(path).convert("L")
    arr = np.asarray(mask, dtype=np.uint8)
    rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    rgb[..., 1] = arr
    rgb[..., 2] = arr
    return Image.fromarray(rgb, mode="RGB")


def outline_from_mask(mask_l: Image.Image) -> np.ndarray:
    arr = np.asarray(mask_l, dtype=np.uint8) > 127
    pad = np.pad(arr, 1, mode="constant")
    eroded = (
        pad[1:-1,1:-1] & pad[:-2,1:-1] & pad[2:,1:-1] & pad[1:-1,:-2] & pad[1:-1,2:]
    )
    border = arr & (~eroded)
    return border


def overlay_mask(base: Image.Image, mask_l: Image.Image, color=(0, 255, 255), alpha=90) -> Image.Image:
    base = base.convert("RGBA")
    mask_l = mask_l.convert("L")

    # важно: размеры mask и base должны совпадать
    if mask_l.size != base.size:
        mask_l = mask_l.resize(base.size, Image.NEAREST)

    mask = np.asarray(mask_l, dtype=np.uint8) > 127
    overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    overlay[mask] = [color[0], color[1], color[2], alpha]

    out = Image.alpha_composite(base, Image.fromarray(overlay, mode="RGBA"))

    border = outline_from_mask(mask_l)
    out_arr = np.asarray(out).copy()
    out_arr[border] = [255, 255, 0, 255]

    return Image.fromarray(out_arr, mode="RGBA").convert("RGB")


def uncertainty_to_heatmap(arr: np.ndarray, size_hw) -> Image.Image:
    arr = np.clip(arr.astype(np.float32), 0.0, 1.0)
    if arr.max() > 0:
        arr = arr / arr.max()
    h, w = size_hw
    arr_u8 = (arr * 255).astype(np.uint8)
    img = Image.fromarray(arr_u8, mode="L").resize((w, h), Image.BILINEAR)
    a = np.asarray(img, dtype=np.uint8)
    rgb = np.zeros((a.shape[0], a.shape[1], 3), dtype=np.uint8)
    rgb[..., 0] = a
    rgb[..., 1] = np.minimum(a * 2, 255)
    rgb[..., 2] = 255 - a
    return Image.fromarray(rgb, mode="RGB")


def tile_with_title(img: Image.Image, title: str, tile_size: int, font) -> Image.Image:
    img = img.copy().convert("RGB")
    img.thumbnail((tile_size, tile_size), Image.BILINEAR)
    canvas = Image.new("RGB", (tile_size, tile_size + 40), (255,255,255))
    x = (tile_size - img.width)//2
    y = (tile_size - img.height)//2 + 24
    canvas.paste(img, (x,y))
    d = ImageDraw.Draw(canvas)
    d.text((12,8), title, fill=(0,0,0), font=font)
    return canvas


def make_panel(case_id: str, method_name: str, input_img: Image.Image, annot_img: Image.Image, mask_img: Image.Image, unc_img: Image.Image, footer: List[str]) -> Image.Image:
    font = ImageFont.load_default()
    tile_size = 420
    overlay_input = overlay_mask(input_img, mask_img.convert("L"))
    overlay_annot = overlay_mask(annot_img, mask_img.convert("L"))
    tiles = [
        tile_with_title(input_img, "Input image", tile_size, font),
        tile_with_title(annot_img, "Annotation", tile_size, font),
        tile_with_title(mask_img, "Predicted mask", tile_size, font),
        tile_with_title(overlay_input, "Input + prediction", tile_size, font),
        tile_with_title(overlay_annot, "Annotation + prediction", tile_size, font),
        tile_with_title(unc_img, "Uncertainty heatmap", tile_size, font),
    ]
    pad = 20
    cols = 3
    rows = 2
    panel_w = cols*tile_size + (cols+1)*pad
    panel_h = rows*(tile_size+40) + (rows+1)*pad + 90
    panel = Image.new("RGB", (panel_w, panel_h), (245,245,245))
    d = ImageDraw.Draw(panel)
    d.text((pad, 8), f"Reiss review panel — {case_id} — {method_name}", fill=(0,0,0), font=font)
    for idx, tile in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        x = pad + c*tile_size + c*pad
        y = 28 + pad + r*(tile_size+40) + r*pad
        panel.paste(tile, (x,y))
    footer_y = panel_h - 54
    d.rectangle([0, footer_y-8, panel_w, panel_h], fill=(235,235,235))
    y = footer_y
    for line in footer[:2]:
        d.text((pad, y), line, fill=(0,0,0), font=font)
        y += 16
    return panel


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    pred_path = Path(args.predictions_manifest).expanduser().resolve()
    reiss_root = Path(args.reiss_root).expanduser().resolve() if args.reiss_root else manifest_path.parent
    pred_root = Path(args.predictions_root).expanduser().resolve() if args.predictions_root else pred_path.parent
    out_dir = Path(args.out_dir).expanduser().resolve()
    ensure_dir(out_dir, overwrite=args.overwrite)
    panels_dir = out_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    manifest = {r['case_id']: r for r in load_rows(manifest_path)}
    preds = {r['case_id']: r for r in load_rows(pred_path)}
    case_ids = sorted(set(manifest) & set(preds))
    if args.limit is not None:
        case_ids = case_ids[:args.limit]

    out_rows = []
    for case_id in case_ids:
        m = manifest[case_id]
        p = preds[case_id]
        input_rel = m.get('image_relpath') or m.get('input_relpath')
        if not input_rel:
            raise KeyError(f"Manifest row for case {case_id} has neither image_relpath nor input_relpath")
        input_path = reiss_root / input_rel
        annot_path = reiss_root / m['annotation_relpath']
        mask_path = pred_root / p['mask_512_relpath']
        unc_path = pred_root / p['uncertainty_256_npy_relpath']
        meta_path = pred_root / p['case_output_relpath'] / 'meta.json'
        if not input_path.is_file(): raise FileNotFoundError(input_path)
        if not annot_path.is_file(): raise FileNotFoundError(annot_path)
        if not mask_path.is_file(): raise FileNotFoundError(mask_path)
        if not unc_path.is_file(): raise FileNotFoundError(unc_path)
        meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.is_file() else {}

        input_img = open_rgb(input_path)
        annot_img = open_rgb(annot_path)
        mask_l = Image.open(mask_path).convert('L')
        mask_rgb = open_mask_rgb(mask_path)
        unc_np = np.load(unc_path)
        unc_img = uncertainty_to_heatmap(unc_np, size_hw=(input_img.height, input_img.width))

        footer = [
            f"thr={meta.get('postprocess',{}).get('threshold','?')}  kernel={meta.get('postprocess',{}).get('kernel_size','?')}  min_area={meta.get('postprocess',{}).get('min_area','?')}",
            f"ddim={meta.get('sampling',{}).get('ddim','?')}  ddim_steps={meta.get('sampling',{}).get('ddim_steps','?')}  n_samples={meta.get('sampling',{}).get('n_samples','?')}  tta={meta.get('sampling',{}).get('tta','?')}",
        ]
        panel = make_panel(case_id, args.method_name, input_img, annot_img, mask_l, unc_img, footer)
        panel_path = panels_dir / f"{case_id}_panel.png"
        panel.save(panel_path)

        out_rows.append({
            'case_id': case_id,
            'panel_relpath': panel_path.relative_to(out_dir).as_posix(),
            'image_relpath': m.get('image_relpath') or m.get('input_relpath') or '',
            'input_relpath': m.get('input_relpath') or m.get('image_relpath') or '',
            'annotation_relpath': m['annotation_relpath'],
            'case_output_relpath': p['case_output_relpath'],
            'ch_columns_json': m.get('ch_columns_json','[]'),
            'fil_columns_json': m.get('fil_columns_json','[]'),
            'has_fil_qs': m.get('has_fil_qs','true'),
            'n_ch': m.get('n_ch','0'),
            'n_fil': m.get('n_fil','0'),
        })

    panel_manifest = out_dir / 'panel_manifest.csv'
    fieldnames = ['case_id','panel_relpath','image_relpath','input_relpath','annotation_relpath','case_output_relpath','ch_columns_json','fil_columns_json','has_fil_qs','n_ch','n_fil']
    with panel_manifest.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(out_rows)

    summary = {
        'method_name': args.method_name,
        'manifest_path': str(manifest_path),
        'predictions_manifest_path': str(pred_path),
        'n_panels': len(out_rows),
        'panel_manifest_path': str(panel_manifest),
    }
    (out_dir / 'panel_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f"Built {len(out_rows)} panels -> {panel_manifest}")


if __name__ == '__main__':
    main()

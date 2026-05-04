import os
import csv
from typing import List, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

try:
    import albumentations as A
except Exception:
    A = None


def _list_pngs(dir_path: str) -> List[str]:
    if not os.path.isdir(dir_path):
        return []
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.lower().endswith(".png")]
    return sorted(files)


def _basename_no_ext(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]


def _pair_by_basename(imgs: List[str], masks: List[str]) -> List[Tuple[str, str]]:
    mask_map: Dict[str, str] = {_basename_no_ext(m): m for m in masks}
    pairs: List[Tuple[str, str]] = []

    for img in imgs:
        key = _basename_no_ext(img)
        m = mask_map.get(key)
        if m is not None:
            pairs.append((img, m))

    if len(pairs) == 0:
        n = min(len(imgs), len(masks))
        pairs = list(zip(sorted(imgs)[:n], sorted(masks)[:n]))

    return pairs


def build_default_augmentations():
    if A is None:
        return None
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ])


def build_clean_augmentations():
    if A is None:
        return None

    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=45, p=0.5, border_mode=1),  # cv2.BORDER_REPLICATE == 1
        A.RandomGamma(gamma_limit=(100, 120), p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=(0.0, 0.15),
            contrast_limit=(0.0, 0.2),
            p=0.5
        ),
    ])

def build_center_limb_mask(h: int, w: int, radius_ratio: float = 0.465) -> np.ndarray:
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    r = min(h, w) * radius_ratio

    yy, xx = np.ogrid[:h, :w]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    mask = (dist2 <= r * r).astype(np.float32)
    return mask


class SCSSFolderDataset(Dataset):

    def __init__(self, imgs_dir: str, masks_dir: str, img_size: int = 256, augmentations=None):
        self.imgs = _list_pngs(imgs_dir)
        self.masks = _list_pngs(masks_dir)

        self.pairs = _pair_by_basename(self.imgs, self.masks)
        if len(self.pairs) == 0:
            raise FileNotFoundError(
                f"No paired PNGs found. Check paths:\n  imgs_dir={imgs_dir}\n  masks_dir={masks_dir}"
            )

        self.img_size = img_size
        self.aug = augmentations

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]
        name = os.path.splitext(os.path.basename(img_path))[0]

        img = Image.open(img_path).convert("L").resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize((self.img_size, self.img_size), Image.NEAREST)

        img_np = np.asarray(img, dtype=np.float32) / 255.0
        mask_np = np.asarray(mask, dtype=np.float32) / 255.0
        mask_np = (mask_np > 0.5).astype(np.float32)

        if self.aug is not None:
            augmented = self.aug(image=img_np[..., None], mask=mask_np[..., None])
            img_np = augmented["image"][..., 0]
            mask_np = augmented["mask"][..., 0]

        img_t = torch.from_numpy(img_np)[None, ...]
        mask_t = torch.from_numpy(mask_np)[None, ...]

        return {"image": img_t, "mask": mask_t, "name": name}


class ManifestSegmentationDataset(Dataset):

    def __init__(
        self,
        manifest_csv: str,
        split: str,
        img_size: int = 256,
        augmentations=None,
        source_filter: str = "all"
    ):
        if not os.path.isfile(manifest_csv):
            raise FileNotFoundError(f"manifest not found: {manifest_csv}")

        self.img_size = img_size
        self.aug = augmentations
        self.records = []

        with open(manifest_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] != split:
                    continue
                if source_filter != "all" and row["source"] != source_filter:
                    continue
                self.records.append(row)

        if len(self.records) == 0:
            raise RuntimeError(
                f"No records found for split='{split}', source_filter='{source_filter}' in {manifest_csv}"
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        row = self.records[idx]

        img_path = row["image_path"]
        mask_path = row["mask_path"]
        name = row["sample_id"]

        img = Image.open(img_path).convert("L").resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize((self.img_size, self.img_size), Image.NEAREST)

        img_np = np.asarray(img, dtype=np.float32) / 255.0

        mask_np = np.asarray(mask, dtype=np.uint8)
        mask_np = (mask_np > 127).astype(np.float32)

        disk_mask = build_center_limb_mask(self.img_size, self.img_size, radius_ratio=0.465)

        img_np = img_np * disk_mask + (1.0 - disk_mask) * 1.0
        mask_np = mask_np * disk_mask

        if self.aug is not None:
            augmented = self.aug(image=img_np[..., None], mask=mask_np[..., None])
            img_np = augmented["image"][..., 0]
            mask_np = augmented["mask"][..., 0]

        img_t = torch.from_numpy(img_np)[None, ...]
        mask_t = torch.from_numpy(mask_np)[None, ...]
        disk_t = torch.from_numpy(disk_mask.astype(np.float32))[None, ...]

        return {
            "image": img_t,
            "mask": mask_t,
            "disk_mask": disk_t,
            "name": name,
            "source": row["source"],
            "timestamp_group": row["timestamp_group"],
        }
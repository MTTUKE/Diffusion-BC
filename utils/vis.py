import os
from typing import Optional
import numpy as np
from PIL import Image

def _to_uint8(x01: np.ndarray) -> np.ndarray:
    x01 = np.clip(x01, 0.0, 1.0)
    return (x01 * 255.0).astype(np.uint8)

def save_triplet(
    out_path: str,
    img01: np.ndarray,
    gt01: np.ndarray,
    pred01: np.ndarray,
    unc01: Optional[np.ndarray] = None
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    parts = [
        Image.fromarray(_to_uint8(img01)).convert("L"),
        Image.fromarray(_to_uint8(gt01)).convert("L"),
        Image.fromarray(_to_uint8(pred01)).convert("L"),
    ]
    if unc01 is not None:
        parts.append(Image.fromarray(_to_uint8(unc01)).convert("L"))

    w, h = parts[0].size
    canvas = Image.new("L", (w * len(parts), h))

    for i, p in enumerate(parts):
        canvas.paste(p, (i * w, 0))

    canvas.save(out_path)
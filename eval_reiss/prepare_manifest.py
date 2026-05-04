from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

INPUT_DIR_NAME = "input_png"
ANNOT_DIR_NAME = "labels_png"

INPUT_TS_RE = re.compile(r"(\d{4}[\-_]\d{2}[\-_]\d{2}T\d{2}[\-_]\d{2}[\-_]\d{2})", re.IGNORECASE)
ANNOT_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}(?:\.\d+)?Z?)", re.IGNORECASE)

INPUT_PATTERNS = [
    "%Y_%m_%dT%H_%M_%S",
    "%Y-%m-%dT%H_%M_%S",
    "%Y_%m_%d_%H_%M_%S",
    "%Y-%m-%d_%H_%M_%S",
]
ANNOT_PATTERNS = [
    "%Y-%m-%dT%H_%M_%S.%fZ",
    "%Y-%m-%dT%H_%M_%S.%f",
    "%Y-%m-%dT%H_%M_%SZ",
    "%Y-%m-%dT%H_%M_%S",
]

@dataclass(frozen=True)
class InputItem:
    path: Path
    timestamp: datetime

@dataclass(frozen=True)
class AnnotationItem:
    path: Path
    timestamp: datetime

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare manifest for local comparison_29 benchmark.")
    p.add_argument("--source", type=Path, required=True, help="Path to comparison_29 folder.")
    p.add_argument("--out-dir", type=Path, required=True, help="Where to save normalized dataset.")
    p.add_argument("--schema-json", type=Path, default=None, help="Optional reiss_case_schema.json")
    p.add_argument("--max-time-diff-seconds", type=float, default=180.0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--strict", action="store_true")
    return p.parse_args()

def ensure_out_dir(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "annotations").mkdir(parents=True, exist_ok=True)

def try_parse_datetime(text: str, patterns: List[str]) -> Optional[datetime]:
    for pattern in patterns:
        try:
            return datetime.strptime(text, pattern)
        except ValueError:
            continue
    return None

def extract_input_timestamp(path: Path) -> Optional[datetime]:
    stem = path.stem
    match = INPUT_TS_RE.search(stem)
    if match:
        candidate = match.group(1).replace("-", "_", 2)
        dt = try_parse_datetime(candidate, INPUT_PATTERNS)
        if dt is not None:
            return dt
    parts = stem.split("_")
    candidates = [stem]
    for n in (4, 5, 6, 7):
        if len(parts) >= n:
            candidates.append("_".join(parts[-n:]))
    seen = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        dt = try_parse_datetime(cand, INPUT_PATTERNS)
        if dt is not None:
            return dt
    return None

def extract_annotation_timestamp(path: Path) -> Optional[datetime]:
    stem = path.stem
    match = ANNOT_TS_RE.search(stem)
    if not match:
        return None
    return try_parse_datetime(match.group(1), ANNOT_PATTERNS)

def scan_inputs(input_dir: Path) -> List[InputItem]:
    items: List[InputItem] = []
    skipped = []
    for path in sorted(input_dir.glob("*.png")):
        ts = extract_input_timestamp(path)
        if ts is None:
            skipped.append(path.name)
            continue
        items.append(InputItem(path=path, timestamp=ts))
    if skipped:
        print(f"[prepare_manifest] Skipped {len(skipped)} input files with unsupported names.")
    return items

def scan_annotations(annot_dir: Path) -> List[AnnotationItem]:
    items: List[AnnotationItem] = []
    skipped = []
    for path in sorted(annot_dir.glob("*.png")):
        ts = extract_annotation_timestamp(path)
        if ts is None:
            skipped.append(path.name)
            continue
        items.append(AnnotationItem(path=path, timestamp=ts))
    if skipped:
        print("[prepare_manifest] Skipped annotation files that do not contain a parseable timestamp:")
        for name in skipped[:10]:
            print(f"  - {name}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")
    return items

def load_schema(path: Optional[Path]) -> Tuple[Dict[str, Dict], List[Tuple[datetime, str, Dict]]]:
    if not path:
        return {}, []
    if not path.is_file():
        raise FileNotFoundError(f"Schema JSON not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    items: List[Tuple[datetime, str, Dict]] = []
    by_key: Dict[str, Dict] = {}
    for key, value in data.items():
        dt = try_parse_datetime(key, ANNOT_PATTERNS)
        if dt is None:
            continue
        by_key[key] = value
        items.append((dt, key, value))
    return by_key, items

def find_best_match(inp: InputItem, annots: List[AnnotationItem], max_diff: float) -> Optional[Tuple[AnnotationItem, float]]:
    best_item = None
    best_diff = None
    for annot in annots:
        diff = abs((annot.timestamp - inp.timestamp).total_seconds())
        if best_diff is None or diff < best_diff:
            best_item = annot
            best_diff = diff
    if best_item is None or best_diff is None or best_diff > max_diff:
        return None
    return best_item, best_diff

def image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as img:
        return img.size

def schema_for_annotation(ts: datetime, schema_items: List[Tuple[datetime, str, Dict]], max_diff: float = 1.5) -> Tuple[Optional[str], Dict]:
    best = None
    best_diff = None
    for sdt, key, value in schema_items:
        diff = abs((sdt - ts).total_seconds())
        if best_diff is None or diff < best_diff:
            best = (key, value)
            best_diff = diff
    if best is None or best_diff is None or best_diff > max_diff:
        return None, {}
    return best

def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    input_dir = source / INPUT_DIR_NAME
    annot_dir = source / ANNOT_DIR_NAME
    out_dir = args.out_dir.resolve()

    if not source.exists():
        raise FileNotFoundError(f"Source folder not found: {source}")
    if not input_dir.exists():
        raise FileNotFoundError(f"Missing input folder: {input_dir}")
    if not annot_dir.exists():
        raise FileNotFoundError(f"Missing annotation folder: {annot_dir}")

    ensure_out_dir(out_dir, args.overwrite)
    inputs = scan_inputs(input_dir)
    annots = scan_annotations(annot_dir)
    if not inputs:
        raise RuntimeError(f"No input PNG files found in: {input_dir}")
    if not annots:
        raise RuntimeError(f"No annotation PNG files found in: {annot_dir}")

    schema_by_key, schema_items = load_schema(args.schema_json.resolve() if args.schema_json else None)

    rows = []
    unmatched_inputs = []
    used_annotation_names = set()
    schema_matched = 0

    for idx, inp in enumerate(inputs, start=1):
        match = find_best_match(inp, annots, args.max_time_diff_seconds)
        if match is None:
            unmatched_inputs.append(inp.path.name)
            continue

        annot, diff_seconds = match
        used_annotation_names.add(annot.path.name)

        schema_key, schema_entry = schema_for_annotation(annot.timestamp, schema_items)
        if schema_key:
            schema_matched += 1
            case_id = schema_key
        else:
            case_id = f"case_{idx:02d}"

        safe_case_id = case_id.replace(":", "_")
        image_dst = out_dir / "images" / f"{safe_case_id}.png"
        annot_dst = out_dir / "annotations" / f"{safe_case_id}_annot.png"

        shutil.copy2(inp.path, image_dst)
        shutil.copy2(annot.path, annot_dst)

        img_w, img_h = image_size(image_dst)
        ann_w, ann_h = image_size(annot_dst)

        rows.append({
            "case_id": case_id,
            "schema_case_key": schema_key or "",
            "input_name": inp.path.name,
            "annotation_name": annot.path.name,
            "input_relpath": str(Path("images") / image_dst.name).replace("\\", "/"),
            "image_relpath": str(Path("images") / image_dst.name).replace("\\", "/"),
            "annotation_relpath": str(Path("annotations") / annot_dst.name).replace("\\", "/"),
            "input_timestamp": inp.timestamp.isoformat(),
            "annotation_timestamp": annot.timestamp.isoformat(),
            "time_diff_seconds": f"{diff_seconds:.3f}",
            "image_width": str(img_w),
            "image_height": str(img_h),
            "annotation_width": str(ann_w),
            "annotation_height": str(ann_h),
            "ch_columns_json": json.dumps(schema_entry.get("ch_columns", []), ensure_ascii=False),
            "fil_columns_json": json.dumps(schema_entry.get("fil_columns", []), ensure_ascii=False),
            "has_fil_qs": str(bool(schema_entry.get("has_fil_qs", True))).lower(),
            "n_ch": str(int(schema_entry.get("n_ch", len(schema_entry.get("ch_columns", []))))),
            "n_fil": str(int(schema_entry.get("n_fil", len(schema_entry.get("fil_columns", []))))),
        })

    unused_annotations = sorted({a.path.name for a in annots} - used_annotation_names)

    fieldnames = [
        "case_id",
        "schema_case_key",
        "input_name",
        "annotation_name",
        "input_relpath",
        "image_relpath",
        "annotation_relpath",
        "input_timestamp",
        "annotation_timestamp",
        "time_diff_seconds",
        "image_width",
        "image_height",
        "annotation_width",
        "annotation_height",
        "ch_columns_json",
        "fil_columns_json",
        "has_fil_qs",
        "n_ch",
        "n_fil",
    ]
    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    summary = {
        "source": str(source),
        "input_dir": str(input_dir),
        "annotation_dir": str(annot_dir),
        "schema_json_argument": str(args.schema_json.resolve()) if args.schema_json else None,
        "total_inputs_found": len(inputs),
        "total_annotations_found": len(annots),
        "matched_cases": len(rows),
        "schema_matched_cases": schema_matched,
        "unmatched_inputs": unmatched_inputs,
        "unused_annotations": unused_annotations,
        "max_time_diff_seconds": args.max_time_diff_seconds,
        "notes": [
            "This benchmark version contains visual annotation PNG files only.",
            "Object-level CH/Fil evaluation is completed later during manual review.",
            "Both input_relpath and image_relpath are written for compatibility.",
            "Annotation filename typos such as '-annoot' are tolerated if the timestamp is parseable.",
        ],
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.strict and (unmatched_inputs or unused_annotations):
        raise RuntimeError(
            "Strict mode failed. "
            f"unmatched_inputs={len(unmatched_inputs)}, unused_annotations={len(unused_annotations)}"
        )

    print(f"Prepared dataset in: {out_dir}")
    print(f"Matched cases: {len(rows)}")
    print(f"Unmatched inputs: {len(unmatched_inputs)}")
    print(f"Unused annotations: {len(unused_annotations)}")
    print(f"Schema-matched cases: {schema_matched}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary:  {summary_path}")

if __name__ == "__main__":
    main()

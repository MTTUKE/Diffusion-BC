import os
import csv
import json
import random
import argparse
from typing import Dict, List, Tuple


IGNORE_NAMES = {"thumbs.db", ".ds_store"}


def is_valid_png(filename: str) -> bool:
    name = filename.lower()
    if name in IGNORE_NAMES:
        return False
    if ".ipynb_checkpoints" in name:
        return False
    return name.endswith(".png")


def parse_source_and_group(filename: str) -> Tuple[str, str]:
    base = os.path.splitext(os.path.basename(filename))[0].lower()

    if base.startswith("aia_ch_"):
        return "aia", base[len("aia_ch_"):]
    if base.startswith("suvi_ch_"):
        return "suvi", base[len("suvi_ch_"):]

    raise ValueError(f"Unexpected filename format: {filename}")


def collect_pngs(folder: str) -> Dict[str, str]:
    files = {}
    for name in sorted(os.listdir(folder)):
        if not is_valid_png(name):
            continue
        full = os.path.join(folder, name)
        if os.path.isfile(full):
            files[name] = full
    return files


def make_group_split(groups: List[str], seed: int, train_ratio: float, val_ratio: float):
    rnd = random.Random(seed)
    groups = sorted(groups)
    rnd.shuffle(groups)

    n = len(groups)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_groups = groups[:n_train]
    val_groups = groups[n_train:n_train + n_val]
    test_groups = groups[n_train + n_val:n_train + n_val + n_test]

    group_to_split = {}
    for g in train_groups:
        group_to_split[g] = "train"
    for g in val_groups:
        group_to_split[g] = "val"
    for g in test_groups:
        group_to_split[g] = "test"

    return {
        "seed": seed,
        "train_groups": train_groups,
        "val_groups": val_groups,
        "test_groups": test_groups,
        "group_to_split": group_to_split,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--masks_dir", type=str, required=True)
    parser.add_argument("--manifest_out", type=str, required=True)
    parser.add_argument("--split_out", type=str, required=True)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio <= 0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    if not os.path.isdir(args.images_dir):
        raise FileNotFoundError(args.images_dir)
    if not os.path.isdir(args.masks_dir):
        raise FileNotFoundError(args.masks_dir)

    imgs = collect_pngs(args.images_dir)
    masks = collect_pngs(args.masks_dir)

    common_names = sorted(set(imgs.keys()) & set(masks.keys()))
    if len(common_names) == 0:
        raise RuntimeError("No matching image/mask filenames found.")

    records = []
    all_groups = set()
    source_count = {"aia": 0, "suvi": 0}

    for name in common_names:
        source, timestamp_group = parse_source_and_group(name)
        all_groups.add(timestamp_group)
        source_count[source] += 1

        sample_id = os.path.splitext(name)[0]

        records.append({
            "sample_id": sample_id,
            "image_path": imgs[name].replace("\\", "/"),
            "mask_path": masks[name].replace("\\", "/"),
            "source": source,
            "timestamp_group": timestamp_group,
        })

    split_info = make_group_split(
        groups=list(all_groups),
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )

    for r in records:
        r["split"] = split_info["group_to_split"][r["timestamp_group"]]

    os.makedirs(os.path.dirname(args.manifest_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.split_out), exist_ok=True)

    with open(args.manifest_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "image_path", "mask_path", "source", "timestamp_group", "split"]
        )
        writer.writeheader()
        writer.writerows(records)

    split_json = {
        "seed": split_info["seed"],
        "train_groups": split_info["train_groups"],
        "val_groups": split_info["val_groups"],
        "test_groups": split_info["test_groups"],
    }

    with open(args.split_out, "w", encoding="utf-8") as f:
        json.dump(split_json, f, indent=2)

    split_counts = {"train": 0, "val": 0, "test": 0}
    for r in records:
        split_counts[r["split"]] += 1

    print("Done.")
    print(f"Matched pairs: {len(records)}")
    print(f"AIA: {source_count['aia']}")
    print(f"SUVI: {source_count['suvi']}")
    print(f"Unique timestamp groups: {len(all_groups)}")
    print(f"Split counts: {split_counts}")
    print(f"Manifest saved to: {args.manifest_out}")
    print(f"Split saved to: {args.split_out}")


if __name__ == "__main__":
    main()
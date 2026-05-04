from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive object-level review for local comparison_29 Reiss benchmark.")
    p.add_argument("--panel-manifest", required=True)
    p.add_argument("--panels-root", default=None, help="Default: parent of panel_manifest.csv")
    p.add_argument("--review-dir", required=True)
    p.add_argument("--method-name", default="Diffusion-BC")
    p.add_argument("--case-id", default=None)
    p.add_argument("--start-case", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--open-panel", action="store_true")
    p.add_argument("--sync-only", action="store_true", help="Rebuild review_manifest.csv from existing case JSON files")
    return p.parse_args()


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open('r', newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"CSV is empty: {path}")
    return rows


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def open_path_in_viewer(path: Path) -> None:
    try:
        if sys.platform.startswith('win'):
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', str(path)])
        else:
            subprocess.Popen(['xdg-open', str(path)])
    except Exception as e:
        print(f"Could not open panel automatically: {e}")


def parse_json_list(text: str) -> List[str]:
    if not text:
        return []
    try:
        data = json.loads(text)
        return list(data) if isinstance(data, list) else []
    except Exception:
        return []


def parse_bool(text: str, default=True) -> bool:
    if text is None:
        return default
    return str(text).strip().lower() in {'1','true','yes','y'}


def normalize_binary(token: str) -> Optional[int]:
    token = token.strip().lower()
    if token in {'0','n','no','false'}:
        return 0
    if token in {'1','y','yes','true'}:
        return 1
    return None


def prompt_group(name: str, keys: List[str], current: Dict[str, int]) -> Dict[str, int]:
    if not keys:
        return {}
    cur = ', '.join(f"{k}={current.get(k,0)}" for k in keys)
    while True:
        raw = input(f"{name} [{cur}] -> enter {len(keys)} values 0/1 separated by comma (Enter = keep): ").strip()
        if raw == '':
            return {k: int(current.get(k, 0)) for k in keys}
        parts = [p.strip() for p in raw.replace(';',',').split(',') if p.strip()!='']
        if len(parts) != len(keys):
            print(f"Need exactly {len(keys)} values.")
            continue
        vals=[]
        ok=True
        for p in parts:
            b = normalize_binary(p)
            if b is None:
                ok=False
                break
            vals.append(b)
        if not ok:
            print("Use only 0/1 or yes/no.")
            continue
        return {k:v for k,v in zip(keys, vals)}


def prompt_single(label: str, current: int) -> int:
    while True:
        raw = input(f"{label} [{current}] -> 0/1 (Enter = keep): ").strip()
        if raw == '':
            return int(current)
        b = normalize_binary(raw)
        if b is not None:
            return b
        print('Use 0 or 1.')


def prompt_int(label: str, current: int) -> int:
    while True:
        raw = input(f"{label} [{current}] -> integer (Enter = keep): ").strip()
        if raw == '':
            return int(current)
        try:
            return int(raw)
        except ValueError:
            print('Enter an integer.')


def save_case_json(path: Path, data: Dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def load_case_json(path: Path) -> Optional[Dict]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def compute_totals(ch: Dict[str,int], fil: Dict[str,int], fil_qs_present: bool, fil_qs_value: int) -> Dict[str,int]:
    ch_pos = sum(int(v) for v in ch.values())
    ch_neg = len(ch) - ch_pos
    fil_vals = [int(v) for v in fil.values()]
    if fil_qs_present:
        fil_vals.append(int(fil_qs_value))
    fil_pos = sum(fil_vals)
    fil_neg = len(fil_vals) - fil_pos
    return {'CH_pos': ch_pos, 'CH_neg': ch_neg, 'Fil_pos': fil_pos, 'Fil_neg': fil_neg}


def write_review_manifest(review_dir: Path, rows: List[Dict[str, object]]) -> Path:
    path = review_dir / 'review_manifest.csv'
    fieldnames = [
        'case_id','panel_relpath','image_relpath','input_relpath','annotation_relpath','review_json_relpath','reviewed',
        'ch_columns_json','fil_columns_json','has_fil_qs','CH_pos','CH_neg','Fil_pos','Fil_neg','other_fp_count','notes'
    ]
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)
    return path


def main() -> None:
    args = parse_args()
    panel_manifest_path = Path(args.panel_manifest).expanduser().resolve()
    panels_root = Path(args.panels_root).expanduser().resolve() if args.panels_root else panel_manifest_path.parent
    review_dir = Path(args.review_dir).expanduser().resolve()
    cases_dir = review_dir / 'cases'
    ensure_dir(cases_dir)

    panel_rows = load_rows(panel_manifest_path)
    panel_rows.sort(key=lambda r: r['case_id'])
    if args.case_id:
        panel_rows = [r for r in panel_rows if r['case_id'] == args.case_id]
    if args.start_case:
        panel_rows = [r for r in panel_rows if r['case_id'] >= args.start_case]
    if args.limit is not None:
        panel_rows = panel_rows[:args.limit]

    collected = []
    for row in panel_rows:
        case_id = row['case_id']
        review_json_path = cases_dir / f"{case_id}.json"
        existing = load_case_json(review_json_path)
        if args.resume and existing and not args.sync_only:
            print(f"Skipping already reviewed case: {case_id}")
            data = existing
        else:
            if args.sync_only:
                data = existing
                if data is None:
                    print(f"No saved review JSON for {case_id}, skipping in sync-only mode.")
                    continue
            else:
                ch_cols = parse_json_list(row.get('ch_columns_json','[]'))
                fil_cols = parse_json_list(row.get('fil_columns_json','[]'))
                has_fil_qs = parse_bool(row.get('has_fil_qs','true'), default=True)
                cur_ch = (existing or {}).get('ch_objects', {})
                cur_fil = (existing or {}).get('fil_objects', {})
                cur_fil_qs = int((existing or {}).get('fil_qs', 0))
                cur_other = int((existing or {}).get('other_fp_count', 0))
                cur_notes = str((existing or {}).get('notes', ''))

                panel_path = panels_root / row['panel_relpath']
                print("\n" + "="*80)
                print(f"Case: {case_id}")
                print(f"Panel: {panel_path}")
                if args.open_panel and panel_path.is_file():
                    open_path_in_viewer(panel_path)
                print(f"CH objects : {', '.join(ch_cols) if ch_cols else '(none)'}")
                print(f"Fil objects: {', '.join(fil_cols) if fil_cols else '(none)'}")
                print(f"Fil_QS     : {'yes' if has_fil_qs else 'no'}")

                ch_data = prompt_group('CH', ch_cols, {k:int(cur_ch.get(k,0)) for k in ch_cols})
                fil_data = prompt_group('Fil', fil_cols, {k:int(cur_fil.get(k,0)) for k in fil_cols})
                fil_qs_value = prompt_single('Fil_QS', cur_fil_qs) if has_fil_qs else 0
                other_fp_count = prompt_int('Other false-positive object count', cur_other)
                notes = input(f"Notes [{cur_notes}] -> Enter text (Enter = keep): ").strip()
                if notes == '':
                    notes = cur_notes

                totals = compute_totals(ch_data, fil_data, has_fil_qs, fil_qs_value)
                data = {
                    'case_id': case_id,
                    'method_name': args.method_name,
                    'panel_relpath': row['panel_relpath'],
                    'image_relpath': row.get('image_relpath') or row.get('input_relpath') or '',
            'input_relpath': row.get('input_relpath') or row.get('image_relpath') or '',
                    'annotation_relpath': row['annotation_relpath'],
                    'ch_columns': ch_cols,
                    'fil_columns': fil_cols,
                    'has_fil_qs': has_fil_qs,
                    'ch_objects': ch_data,
                    'fil_objects': fil_data,
                    'fil_qs': fil_qs_value,
                    'other_fp_count': other_fp_count,
                    'notes': notes,
                    **totals,
                }
                save_case_json(review_json_path, data)

        collected.append({
            'case_id': case_id,
            'panel_relpath': row['panel_relpath'],
            'image_relpath': row.get('image_relpath') or row.get('input_relpath') or '',
            'input_relpath': row.get('input_relpath') or row.get('image_relpath') or '',
            'annotation_relpath': row['annotation_relpath'],
            'review_json_relpath': review_json_path.relative_to(review_dir).as_posix(),
            'reviewed': 'true' if data is not None else 'false',
            'ch_columns_json': json.dumps(data.get('ch_columns', parse_json_list(row.get('ch_columns_json','[]'))), ensure_ascii=False) if data else row.get('ch_columns_json','[]'),
            'fil_columns_json': json.dumps(data.get('fil_columns', parse_json_list(row.get('fil_columns_json','[]'))), ensure_ascii=False) if data else row.get('fil_columns_json','[]'),
            'has_fil_qs': str(bool(data.get('has_fil_qs', parse_bool(row.get('has_fil_qs','true'))))).lower() if data else row.get('has_fil_qs','true'),
            'CH_pos': data.get('CH_pos','') if data else '',
            'CH_neg': data.get('CH_neg','') if data else '',
            'Fil_pos': data.get('Fil_pos','') if data else '',
            'Fil_neg': data.get('Fil_neg','') if data else '',
            'other_fp_count': data.get('other_fp_count','') if data else '',
            'notes': data.get('notes','') if data else '',
        })

    review_manifest = write_review_manifest(review_dir, collected)
    summary = {
        'method_name': args.method_name,
        'panel_manifest': str(panel_manifest_path),
        'review_dir': str(review_dir),
        'n_cases_in_manifest': len(panel_rows),
        'n_review_json_found': sum(1 for r in collected if r['reviewed'] == 'true'),
        'review_manifest_path': str(review_manifest),
    }
    (review_dir / 'review_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f"Saved review manifest -> {review_manifest}")


if __name__ == '__main__':
    main()

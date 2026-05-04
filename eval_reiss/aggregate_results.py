from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate final Reiss results for Diffusion-BC and compare with reference methods.")
    p.add_argument("--review-manifest", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--method-name", default="Diffusion-BC")
    p.add_argument("--reference-summary", default=None, help="Optional reiss_reference_summary.csv bundled with the scripts")
    p.add_argument("--strict", action="store_true", help="Require exactly 29 reviewed cases")
    return p.parse_args()


def infer_reference_path() -> Optional[Path]:
    cand = Path(__file__).resolve().with_name('reiss_reference_summary.csv')
    return cand if cand.is_file() else None


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open('r', newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"CSV is empty: {path}")
    return rows


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict]) -> None:
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def main() -> None:
    args = parse_args()
    review_manifest = Path(args.review_manifest).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(review_manifest)
    reviewed = [r for r in rows if str(r.get('reviewed','')).lower() in {'true','1','yes','y'}]
    if args.strict and len(reviewed) != 29:
        raise RuntimeError(f"Strict mode expects 29 reviewed cases, got {len(reviewed)}")
    if not reviewed:
        raise RuntimeError('No reviewed cases found in review_manifest.csv')

    ch_pos = sum(int(float(r.get('CH_pos') or 0)) for r in reviewed)
    ch_neg = sum(int(float(r.get('CH_neg') or 0)) for r in reviewed)
    fil_pos = sum(int(float(r.get('Fil_pos') or 0)) for r in reviewed)
    fil_neg = sum(int(float(r.get('Fil_neg') or 0)) for r in reviewed)
    other_fp_count = sum(int(float(r.get('other_fp_count') or 0)) for r in reviewed)
    diffusion_row = {
        'method': args.method_name,
        'n_cases': len(reviewed),
        'CH_pos': ch_pos,
        'CH_neg': ch_neg,
        'CH_TPR': round(safe_div(ch_pos, ch_pos + ch_neg), 4),
        'Fil_pos': fil_pos,
        'Fil_neg': fil_neg,
        'Fil_FPR': round(safe_div(fil_pos, fil_pos + fil_neg), 4),
        'other_false_positive_count': other_fp_count,
    }

    ref_path = Path(args.reference_summary).expanduser().resolve() if args.reference_summary else infer_reference_path()
    combined: List[Dict] = []
    if ref_path and ref_path.is_file():
        refs = load_rows(ref_path)
        for r in refs:
            combined.append({
                'method': r['method'],
                'n_cases': int(float(r.get('n_cases') or 0)),
                'CH_pos': int(float(r.get('CH_pos') or 0)),
                'CH_neg': int(float(r.get('CH_neg') or 0)),
                'CH_TPR': float(r.get('CH_TPR') or 0),
                'Fil_pos': int(float(r.get('Fil_pos') or 0)),
                'Fil_neg': int(float(r.get('Fil_neg') or 0)),
                'Fil_FPR': float(r.get('Fil_FPR') or 0),
                'other_false_positive_count': '',
            })
    combined = [r for r in combined if r['method'] != args.method_name]
    combined.append(diffusion_row)
    combined.sort(key=lambda r: (-float(r['CH_TPR']), float(r['Fil_FPR']), str(r['method'])))
    for idx, r in enumerate(combined, start=1):
        r['rank_by_tpr_then_fpr'] = idx

    write_csv(out_dir / 'diffusion_bc_only.csv', list(diffusion_row.keys()), [diffusion_row])
    (out_dir / 'diffusion_bc_only.json').write_text(json.dumps(diffusion_row, indent=2), encoding='utf-8')

    combined_fields = ['rank_by_tpr_then_fpr','method','n_cases','CH_pos','CH_neg','CH_TPR','Fil_pos','Fil_neg','Fil_FPR','other_false_positive_count']
    write_csv(out_dir / 'reiss_summary_with_reference.csv', combined_fields, combined)
    (out_dir / 'reiss_summary_with_reference.json').write_text(json.dumps(combined, indent=2), encoding='utf-8')

    case_fields = ['case_id','CH_pos','CH_neg','Fil_pos','Fil_neg','other_fp_count','notes']
    case_rows = [{k:r.get(k,'') for k in case_fields} for r in reviewed]
    write_csv(out_dir / 'reviewed_cases.csv', case_fields, case_rows)

    thesis_md = []
    thesis_md.append(f"# Reiss benchmark summary for {args.method_name}\n")
    thesis_md.append(f"Reviewed cases: {len(reviewed)}")
    thesis_md.append(f"CH_pos = {ch_pos}, CH_neg = {ch_neg}, CH_TPR = {diffusion_row['CH_TPR']:.4f}")
    thesis_md.append(f"Fil_pos = {fil_pos}, Fil_neg = {fil_neg}, Fil_FPR = {diffusion_row['Fil_FPR']:.4f}")
    thesis_md.append(f"Other false-positive object count = {other_fp_count}")
    if ref_path and ref_path.is_file():
        rank = next((r['rank_by_tpr_then_fpr'] for r in combined if r['method'] == args.method_name), None)
        thesis_md.append(f"Rank in combined reference table (higher TPR, lower FPR): {rank}")
        thesis_md.append("")
        thesis_md.append("Reference comparison uses CH_TPR and Fil_FPR only. The 'other false-positive object count' is stored only for Diffusion-BC in this local pipeline, so it should be discussed as an extra qualitative metric, not as a directly comparable published ranking column.")
    (out_dir / 'reiss_summary_for_thesis.md').write_text('\n'.join(thesis_md) + '\n', encoding='utf-8')

    summary = {
        'method_name': args.method_name,
        'review_manifest': str(review_manifest),
        'reviewed_cases': len(reviewed),
        'reference_summary': str(ref_path) if ref_path and ref_path.is_file() else None,
        'diffusion_row': diffusion_row,
        'output_dir': str(out_dir),
    }
    (out_dir / 'aggregate_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f"Aggregated {len(reviewed)} cases -> {out_dir}")


if __name__ == '__main__':
    main()

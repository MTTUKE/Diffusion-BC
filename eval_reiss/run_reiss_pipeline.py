from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the whole local comparison_29 Reiss pipeline.")
    p.add_argument("--source", required=True, help="Path to comparison_29 folder")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint, e.g. checkpoints/best.pt")
    p.add_argument("--work-dir", required=True, help="Base output folder for all intermediate and final results")
    p.add_argument("--project-root", default=None, help="Project root containing config.py, models/, diffusion/")
    p.add_argument("--scripts-dir", default=None, help="Folder containing these pipeline scripts")
    p.add_argument("--schema-json", default=None, help="Optional explicit path to reiss_case_schema.json")
    p.add_argument("--reference-summary", default=None, help="Optional explicit path to reiss_reference_summary.csv")
    p.add_argument("--method-name", default="Diffusion-BC")
    p.add_argument("--mode", default="all", choices=["all","auto","review","aggregate"])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--open-panel", action="store_true")
    p.add_argument("--resume-review", action="store_true")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-samples", type=int, default=6)
    p.add_argument("--ddim-steps", type=int, default=100)
    p.add_argument("--thr", type=float, default=0.72)
    p.add_argument("--tta", default="flip4", choices=["none","flip4"])
    p.add_argument("--kernel-size", type=int, default=5)
    p.add_argument("--min-area", type=int, default=200)
    return p.parse_args()


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def run(cmd: List[str], cwd: Path) -> None:
    print("\n>>>", ' '.join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    args = parse_args()
    scripts_dir = Path(args.scripts_dir).expanduser().resolve() if args.scripts_dir else Path(__file__).resolve().parent
    project_root = Path(args.project_root).expanduser().resolve() if args.project_root else scripts_dir.parent
    source = Path(args.source).expanduser().resolve()
    ckpt = Path(args.ckpt).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    schema_json = Path(args.schema_json).expanduser().resolve() if args.schema_json else (scripts_dir / 'reiss_case_schema.json')
    reference_summary = Path(args.reference_summary).expanduser().resolve() if args.reference_summary else (scripts_dir / 'reiss_reference_summary.csv')

    require(source); require(ckpt); require(project_root / 'config.py'); require(project_root / 'models'); require(project_root / 'diffusion')
    require(scripts_dir / 'prepare_manifest.py'); require(scripts_dir / 'infer_reiss.py'); require(scripts_dir / 'build_panels.py'); require(scripts_dir / 'review_cases.py'); require(scripts_dir / 'aggregate_results.py')

    dataset_dir = work_dir / 'reiss_dataset'
    outputs_dir = work_dir / 'reiss_outputs'
    panels_dir = work_dir / 'reiss_panels'
    review_dir = work_dir / 'reiss_review'
    aggregate_dir = work_dir / 'reiss_aggregate'

    py = sys.executable

    if args.mode in {'all','auto'}:
        cmd = [py, str(scripts_dir / 'prepare_manifest.py'), '--source', str(source), '--out-dir', str(dataset_dir)]
        if schema_json.is_file():
            cmd += ['--schema-json', str(schema_json)]
        if args.overwrite:
            cmd += ['--overwrite']
        run(cmd, cwd=project_root)

        cmd = [py, str(scripts_dir / 'infer_reiss.py'), '--manifest', str(dataset_dir / 'manifest.csv'), '--ckpt', str(ckpt), '--out-dir', str(outputs_dir), '--method-name', args.method_name, '--ddim', '--ddim-steps', str(args.ddim_steps), '--n-samples', str(args.n_samples), '--thr', str(args.thr), '--tta', args.tta, '--kernel-size', str(args.kernel_size), '--min-area', str(args.min_area), '--seed', str(args.seed)]
        if args.overwrite:
            cmd += ['--overwrite']
        run(cmd, cwd=project_root)

        cmd = [py, str(scripts_dir / 'build_panels.py'), '--manifest', str(dataset_dir / 'manifest.csv'), '--predictions-manifest', str(outputs_dir / 'predictions_manifest.csv'), '--out-dir', str(panels_dir), '--method-name', args.method_name]
        if args.overwrite:
            cmd += ['--overwrite']
        run(cmd, cwd=project_root)

    if args.mode in {'all','review'}:
        cmd = [py, str(scripts_dir / 'review_cases.py'), '--panel-manifest', str(panels_dir / 'panel_manifest.csv'), '--review-dir', str(review_dir), '--method-name', args.method_name]
        if args.open_panel:
            cmd += ['--open-panel']
        if args.resume_review:
            cmd += ['--resume']
        run(cmd, cwd=project_root)

    if args.mode in {'all','aggregate'}:
        cmd = [py, str(scripts_dir / 'aggregate_results.py'), '--review-manifest', str(review_dir / 'review_manifest.csv'), '--out-dir', str(aggregate_dir), '--method-name', args.method_name]
        if reference_summary.is_file():
            cmd += ['--reference-summary', str(reference_summary)]
        if args.strict:
            cmd += ['--strict']
        run(cmd, cwd=project_root)

    print(f"\nDone. Final aggregate folder: {aggregate_dir}")


if __name__ == '__main__':
    main()

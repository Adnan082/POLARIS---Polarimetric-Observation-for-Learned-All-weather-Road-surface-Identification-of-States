"""
One-time precomputation of Stokes parameters for the full PRISM dataset.

Reads raw 4-angle polar PNGs, computes (6, H, W) float32 arrays:
    [S0, S1, S2, DoLP, sin(2·AoLP), cos(2·AoLP)]
and saves them as .npy files. Training loads these instead of recomputing
from 4 PNGs every epoch.

EC2 usage (c5.4xlarge, 16 vCPU):
    python scripts/precompute_stokes.py \
        --data-dir /home/ubuntu/polaris_data \
        --output-dir /home/ubuntu/polaris_stokes \
        --workers 12

After completion, sync to S3:
    aws s3 sync /home/ubuntu/polaris_stokes s3://YOUR_BUCKET/processed/stokes/

Resume-safe: skips frames whose .npy already exists.
Expected time: ~3-5 hrs on c5.4xlarge for full dataset.
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.stokes import compute_stokes, pack_polar_channels
from src.data.utils import POLAR_DIRS, imread_unicode


def process_frame(args: tuple) -> str:
    """Process a single frame — must be a top-level function for multiprocessing."""
    seq_dir, stem, out_path = args

    if out_path.exists():
        return "skip"

    arrays = {}
    for angle, subdir in POLAR_DIRS.items():
        p = Path(seq_dir) / "polar" / subdir / f"{stem}.png"
        arr = imread_unicode(p)
        if arr is None:
            return f"missing:{p}"
        arrays[angle] = arr.astype(np.float32)

    stokes = compute_stokes(arrays["0"], arrays["45"], arrays["90"], arrays["135"])
    packed = pack_polar_channels(stokes)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), packed)
    return "done"


def collect_jobs(data_root: Path, out_root: Path) -> list:
    """Walk all splits/sessions/sequences and return list of (seq_dir, stem, out_path)."""
    jobs = []
    for split in ["train", "val"]:
        split_dir = data_root / split
        if not split_dir.exists():
            continue
        for session_dir in sorted(split_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            for seq_dir in sorted(session_dir.iterdir()):
                if not seq_dir.is_dir() or seq_dir.name == "vehicle_state":
                    continue
                polar_dir = seq_dir / "polar" / "0d"
                if not polar_dir.exists():
                    continue
                out_seq = out_root / split / session_dir.name / seq_dir.name
                for frame_path in sorted(polar_dir.glob("*.png")):
                    stem = frame_path.stem
                    out_path = out_seq / f"{stem}.npy"
                    jobs.append((str(seq_dir), stem, str(out_path)))
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   required=True, help="Root with train/ and val/ dirs")
    parser.add_argument("--output-dir", default=None,  help="Where to write .npy files (default: data-dir/stokes)")
    parser.add_argument("--workers",    type=int, default=8, help="Parallel workers (default 8, use 12 on c5.4xlarge)")
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    out_root  = Path(args.output_dir) if args.output_dir else data_root / "stokes"

    print(f"Data root  : {data_root}")
    print(f"Output root: {out_root}")
    print(f"Workers    : {args.workers}")

    jobs = collect_jobs(data_root, out_root)
    print(f"Total frames: {len(jobs):,}")

    already_done = sum(1 for _, _, out in jobs if Path(out).exists())
    print(f"Already done: {already_done:,}  |  To process: {len(jobs) - already_done:,}")

    if len(jobs) == already_done:
        print("Nothing to do — all frames already precomputed.")
        return

    done = skipped = errors = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_frame, job): job for job in jobs}
        with tqdm(total=len(jobs), unit="frame") as bar:
            for future in as_completed(futures):
                result = future.result()
                if result == "done":
                    done += 1
                elif result == "skip":
                    skipped += 1
                else:
                    errors += 1
                    tqdm.write(f"ERROR: {result}")
                bar.set_postfix(done=done, skip=skipped, err=errors)
                bar.update(1)

    print(f"\nComplete — done={done:,}  skipped={skipped:,}  errors={errors:,}")
    print(f"Output: {out_root}")
    print(f"\nNext step — sync to S3:")
    print(f"  aws s3 sync {out_root} s3://YOUR_BUCKET/processed/stokes/ --quiet")


if __name__ == "__main__":
    main()

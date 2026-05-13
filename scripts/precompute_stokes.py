"""
One-time script: reads raw 4-angle polarization PNGs from local disk,
computes Stokes parameters, writes (5, H, W) float32 .npy files back to S3.

Run on a CPU EC2 instance after transfer_setup.sh completes.
Cost: c5.4xlarge ~$0.68/hr, expect 3-5 hrs for full dataset.

Usage:
    python scripts/precompute_stokes.py --data-dir /mnt/data --bucket polaris-prism
"""

import argparse
import os
from pathlib import Path

import numpy as np
import boto3
from tqdm import tqdm

from src.data.stokes import compute_stokes, pack_polar_channels
from src.data.utils import POLAR_DIRS, imread_unicode


def process_session(session_dir: Path, out_root: Path, s3, bucket: str):
    for seq_dir in sorted(session_dir.iterdir()):
        if not seq_dir.is_dir() or seq_dir.name == "vehicle_state":
            continue

        rgb_dir = seq_dir / "rgb"
        if not rgb_dir.exists():
            continue

        out_seq = out_root / session_dir.name / seq_dir.name
        out_seq.mkdir(parents=True, exist_ok=True)

        frames = sorted(rgb_dir.glob("*.png"))
        for frame_path in tqdm(frames, desc=f"{session_dir.name}/{seq_dir.name}", leave=False):
            stem = frame_path.stem
            out_path = out_seq / f"{stem}.npy"

            if out_path.exists():
                continue    # resume-safe

            arrays = {}
            for angle, subdir in POLAR_DIRS.items():
                p = seq_dir / "polar" / subdir / f"{stem}.png"
                arr = imread_unicode(p)
                if arr is None:
                    print(f"  WARNING: missing {p}, skipping frame")
                    break
                arrays[angle] = arr.astype(np.float32)
            else:
                stokes = compute_stokes(arrays["0"], arrays["45"], arrays["90"], arrays["135"])
                packed = pack_polar_channels(stokes)
                np.save(str(out_path), packed)

                if s3 and bucket:
                    s3_key = f"processed/stokes/{session_dir.name}/{seq_dir.name}/{stem}.npy"
                    s3.upload_file(str(out_path), bucket, s3_key)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="Root dir with train/ and val/ session dirs")
    p.add_argument("--bucket", default=None, help="S3 bucket to upload .npy files (optional)")
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    data_root = Path(args.data_dir)
    out_root  = Path(args.output_dir or data_root / "stokes")
    s3        = boto3.client("s3", region_name="eu-west-2") if args.bucket else None

    for split in ["train", "val"]:
        split_dir = data_root / split
        if not split_dir.exists():
            continue
        sessions = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        print(f"{split}: {len(sessions)} sessions")
        for session_dir in sessions:
            process_session(session_dir, out_root, s3, args.bucket)

    print("Stokes precomputation complete.")


if __name__ == "__main__":
    main()

"""
One-time script: reads raw 4-angle polarization TIFFs from S3,
computes Stokes parameters, writes (5, H, W) float32 .npy files back to S3.

Run on a CPU EC2 instance after transfer_setup.sh completes.
Cost: c5.4xlarge ~$0.68/hr, expect 3-5 hrs for full dataset.

Usage:
    python scripts/precompute_stokes.py --bucket polaris-prism --local-dir /mnt/data
"""

import argparse
import os
from pathlib import Path

import numpy as np
import tifffile
import boto3
from tqdm import tqdm

from src.data.stokes import compute_stokes, pack_polar_channels


ANGLE_DIRS = {"0": "0", "45": "45", "90": "90", "135": "135"}


def load_polar_frame(session_dir: Path, frame_stem: str) -> dict:
    imgs = {}
    for angle, subdir in ANGLE_DIRS.items():
        path = session_dir / "polar" / subdir / f"{frame_stem}.tiff"
        arr = tifffile.imread(str(path)).astype(np.float32)
        imgs[angle] = arr
    return imgs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True)
    p.add_argument("--local-dir", required=True,
                   help="Local dir where raw sessions are synced")
    p.add_argument("--output-dir", default=None,
                   help="Local output dir (default: <local-dir>/stokes)")
    args = p.parse_args()

    local_root = Path(args.local_dir)
    out_root = Path(args.output_dir or local_root / "stokes")
    s3 = boto3.client("s3", region_name="eu-west-2")

    sessions = sorted([d for d in local_root.iterdir() if d.is_dir() and d.name != "stokes"])
    print(f"Processing {len(sessions)} sessions")

    for session_dir in sessions:
        frames = sorted((session_dir / "polar" / "0").glob("*.tiff"))
        out_session = out_root / session_dir.name
        out_session.mkdir(parents=True, exist_ok=True)

        for frame_path in tqdm(frames, desc=session_dir.name, leave=False):
            stem = frame_path.stem
            out_path = out_session / f"{stem}.npy"

            if out_path.exists():
                continue    # resume-safe

            imgs = load_polar_frame(session_dir, stem)
            stokes = compute_stokes(imgs["0"], imgs["45"], imgs["90"], imgs["135"])
            packed = pack_polar_channels(stokes)
            np.save(str(out_path), packed)

            s3_key = f"processed/stokes/{session_dir.name}/{stem}.npy"
            s3.upload_file(str(out_path), args.bucket, s3_key)

    print("Stokes precomputation complete.")


if __name__ == "__main__":
    main()

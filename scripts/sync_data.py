"""
Sync a working subset of PRISM from S3 to local NVMe at the start of a training run.
Only syncs the sessions listed in configs/splits/session_splits.yaml.

Usage:
    python scripts/sync_data.py --bucket polaris-prism --local-dir /mnt/nvme/polaris
    python scripts/sync_data.py --bucket polaris-prism --local-dir /mnt/nvme/polaris --eda-only
"""

import argparse
import subprocess
import yaml
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True)
    p.add_argument("--local-dir", required=True)
    p.add_argument("--splits-file", default="configs/splits/session_splits.yaml")
    p.add_argument("--eda-only", action="store_true",
                   help="Sync only 3 sessions for EDA (one per weather type)")
    p.add_argument("--include-stokes", action="store_true",
                   help="Also sync precomputed Stokes cache from S3")
    return p.parse_args()


def s3_sync(bucket: str, s3_prefix: str, local_path: str, region: str = "eu-west-2"):
    cmd = [
        "aws", "s3", "sync",
        f"s3://{bucket}/{s3_prefix}", local_path,
        "--region", region,
        "--no-progress",
    ]
    print(f"  syncing s3://{bucket}/{s3_prefix} → {local_path}")
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()

    with open(args.splits_file) as f:
        splits = yaml.safe_load(f)

    split_sessions = {
        "train": splits.get("train", []),
        "val":   splits.get("val", []),
    }

    if args.eda_only:
        eda_sessions = splits.get("eda", splits.get("train", [])[:3])
        split_sessions = {"train": eda_sessions, "val": []}
        total = sum(len(v) for v in split_sessions.values())
        print(f"EDA mode: syncing {total} sessions — {eda_sessions}")
    else:
        total = sum(len(v) for v in split_sessions.values())
        print(f"Syncing {total} sessions")

    local_root = Path(args.local_dir)
    local_root.mkdir(parents=True, exist_ok=True)

    for split, sessions in split_sessions.items():
        for session in sessions:
            # S3 layout: raw/train/{session}/ and raw/val/{session}/
            # Local layout mirrors: {local_root}/train/{session}/  (what PRISMDataset expects)
            session_local = str(local_root / split / session)
            s3_sync(args.bucket, f"raw/{split}/{session}", session_local)

            if args.include_stokes:
                stokes_local = str(local_root / "stokes" / split / session)
                s3_sync(args.bucket, f"processed/stokes/{split}/{session}", stokes_local)

    # always sync labels.json
    s3_sync(args.bucket, "raw/labels.json", str(local_root / "labels.json"))

    print("Done.")


if __name__ == "__main__":
    main()

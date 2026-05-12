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

    all_sessions = splits["train"] + splits["val"] + splits["test"]

    if args.eda_only:
        # pick the first 3 sessions — replace with specific sessions after EDA
        sessions = all_sessions[:3]
        print(f"EDA mode: syncing {len(sessions)} sessions only")
    else:
        sessions = all_sessions
        print(f"Syncing {len(sessions)} sessions")

    local_root = Path(args.local_dir)
    local_root.mkdir(parents=True, exist_ok=True)

    for session in sessions:
        session_local = str(local_root / session)
        s3_sync(args.bucket, f"raw/{session}", session_local)

        if args.include_stokes:
            stokes_local = str(local_root / "stokes" / session)
            s3_sync(args.bucket, f"processed/stokes/{session}", stokes_local)

    # always sync splits config
    s3_sync(args.bucket, "splits", str(local_root / "splits"))

    print("Done.")


if __name__ == "__main__":
    main()

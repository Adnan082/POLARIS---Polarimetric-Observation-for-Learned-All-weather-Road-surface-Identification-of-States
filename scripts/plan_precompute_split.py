"""
Query S3 for polar folder sizes per session, then greedily assign sessions
to N instances so each gets roughly equal total data.

Usage:
    python scripts/plan_precompute_split.py --bucket YOUR_BUCKET --instances 4

Output: prints which sessions go to which instance + estimated sizes.
"""

import argparse
import yaml
import boto3
from collections import defaultdict


def get_session_polar_size(s3, bucket: str, split: str, session: str) -> int:
    """Return total bytes of polar images for one session."""
    prefix = f"raw/{split}/{session}/"
    paginator = s3.get_paginator("list_objects_v2")
    total = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if "/polar/" in key and key.endswith(".png"):
                total += obj["Size"]
    return total


def greedy_assign(sessions_sizes: list, n_instances: int) -> list:
    """
    Greedy bin-packing: assign heaviest session to lightest instance.
    Returns list of N lists, each containing (split/session, size) tuples.
    """
    instances = [[] for _ in range(n_instances)]
    totals    = [0] * n_instances

    for session, size in sorted(sessions_sizes, key=lambda x: x[1], reverse=True):
        lightest = totals.index(min(totals))
        instances[lightest].append((session, size))
        totals[lightest] += size

    return instances, totals


def fmt(n_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket",    required=True)
    parser.add_argument("--instances", type=int, default=4)
    parser.add_argument("--splits-file", default="configs/splits/session_splits.yaml")
    parser.add_argument("--region",    default="eu-west-2")
    args = parser.parse_args()

    s3 = boto3.client("s3", region_name=args.region)

    with open(args.splits_file) as f:
        splits = yaml.safe_load(f)

    all_sessions = []
    for split in ["train", "val"]:
        for session in splits.get(split, []):
            all_sessions.append((split, session))

    print(f"Checking {len(all_sessions)} sessions in S3...")
    print(f"Bucket: s3://{args.bucket}/raw/\n")

    sessions_sizes = []
    for split, session in all_sessions:
        size = get_session_polar_size(s3, args.bucket, split, session)
        sessions_sizes.append((f"{split}/{session}", size))
        print(f"  {split}/{session:30s} {fmt(size)}")

    print(f"\n{'='*55}")
    print(f"Total polar data: {fmt(sum(s for _, s in sessions_sizes))}")
    print(f"Splitting across {args.instances} instances...\n")

    instances, totals = greedy_assign(sessions_sizes, args.instances)

    for i, (inst_sessions, total) in enumerate(zip(instances, totals)):
        print(f"Instance {i}  ({fmt(total)}, {len(inst_sessions)} sessions):")
        for session, size in inst_sessions:
            print(f"  {session:40s} {fmt(size)}")
        print()

    print("="*55)
    print("SESSION_LIST for each instance user_data script:")
    for i, inst_sessions in enumerate(instances):
        session_str = " ".join(s for s, _ in inst_sessions)
        print(f"\nInstance {i}:")
        print(f'  SESSION_LIST="{session_str}"')


if __name__ == "__main__":
    main()

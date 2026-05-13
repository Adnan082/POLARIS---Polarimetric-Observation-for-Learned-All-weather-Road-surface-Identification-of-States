#!/usr/bin/env bash
# Run on EC2 in eu-west-2 to pull PRISM from HuggingFace → S3
# Downloads, extracts, uploads, and deletes ONE session at a time
# so disk usage never exceeds ~20 GB regardless of dataset size.
#
# Prerequisites:
#   - IAM role with s3:PutObject on your bucket attached to this instance
#   - HF_TOKEN env var set
#   - S3 bucket already created in eu-west-2
#
# Usage:
#   S3_BUCKET=polaris-prism-xxx HF_TOKEN=hf_xxx bash aws/transfer_setup.sh

set -euo pipefail

S3_BUCKET="${S3_BUCKET:?Set S3_BUCKET env var}"
HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var}"
LOCAL_DIR="/home/ubuntu/prism_data"
HF_REPO="NeurIPS-2026-PRISM/PRISM-Dataset"
LOG_FILE="/home/ubuntu/transfer.log"

echo "=== POLARIS: PRISM Dataset Transfer ===" | tee -a "${LOG_FILE}"
echo "S3 bucket : s3://${S3_BUCKET}/raw/"      | tee -a "${LOG_FILE}"
echo "Started   : $(date)"                      | tee -a "${LOG_FILE}"

# --- system setup ---
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip unzip curl

# Install AWS CLI v2 (not available via apt on Ubuntu 24.04)
if ! command -v aws &> /dev/null; then
    curl -s "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/tmp/awscliv2.zip"
    unzip -q /tmp/awscliv2.zip -d /tmp
    sudo /tmp/aws/install
    rm -rf /tmp/awscliv2.zip /tmp/aws
fi

pip install --quiet --break-system-packages huggingface_hub
mkdir -p "${LOCAL_DIR}"

# --- Step 1: labels.json first ---
echo "Downloading labels.json ..." | tee -a "${LOG_FILE}"
python3 - <<PYEOF
import os
from huggingface_hub import hf_hub_download
import shutil

path = hf_hub_download(
    repo_id=os.environ["HF_REPO"],
    repo_type="dataset",
    filename="labels.json",
    token=os.environ["HF_TOKEN"],
)
shutil.copy(path, os.path.join(os.environ["LOCAL_DIR"], "labels.json"))
print("labels.json downloaded.")
PYEOF

aws s3 cp "${LOCAL_DIR}/labels.json" "s3://${S3_BUCKET}/raw/labels.json" --region eu-west-2
echo "labels.json uploaded." | tee -a "${LOG_FILE}"

# --- Step 2: one session at a time ---
# Download zip → extract → upload to S3 → delete local → next session
python3 - <<PYEOF
import os, subprocess, zipfile, shutil
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

token    = os.environ["HF_TOKEN"]
repo_id  = os.environ["HF_REPO"]
bucket   = os.environ["S3_BUCKET"]
local    = Path(os.environ["LOCAL_DIR"])
log_file = os.environ["LOG_FILE"]

api  = HfApi()
zips = []
for split in ["train", "val"]:
    items = api.list_repo_tree(repo_id, repo_type="dataset", token=token,
                               path_in_repo=split, recursive=False)
    zips += [f.path for f in items if hasattr(f, "path") and f.path.endswith(".zip")]
zips.sort()
zips.sort()

print(f"Found {len(zips)} session zips")

for i, hf_path in enumerate(zips, 1):
    split        = hf_path.split("/")[0]          # train or val
    session_name = Path(hf_path).stem             # e.g. 0106_dataset
    session_dir  = local / split / session_name

    msg = f"[{i}/{len(zips)}] {hf_path}"
    print(msg)
    with open(log_file, "a") as lf: lf.write(msg + "\n")

    # Download zip
    zip_path = hf_hub_download(
        repo_id=repo_id, repo_type="dataset",
        filename=hf_path, token=token,
    )

    # Extract
    (local / split).mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(local / split)

    # Upload extracted session to S3
    s3_prefix = f"s3://{bucket}/raw/{split}/{session_name}"
    subprocess.run([
        "aws", "s3", "sync", str(session_dir), s3_prefix,
        "--region", "eu-west-2",
        "--no-progress",
        "--storage-class", "STANDARD_IA",
    ], check=True)

    # Delete local to free disk
    shutil.rmtree(str(session_dir))
    print(f"  done — disk freed")

print("All sessions transferred.")
PYEOF

echo "=== Transfer finished: $(date) ===" | tee -a "${LOG_FILE}"
echo "Safe to terminate this instance."

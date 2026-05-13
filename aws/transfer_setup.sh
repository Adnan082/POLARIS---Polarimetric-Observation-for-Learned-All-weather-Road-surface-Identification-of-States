#!/usr/bin/env bash
# Run on a c5n.2xlarge in eu-west-2 to pull PRISM from HuggingFace → S3
# Cost: ~$0.38/hr, expect 10-18 hrs for full 1.6TB
# Never run this on your local machine — 1.6TB download
#
# Prerequisites:
#   - IAM role with s3:PutObject on your bucket attached to this instance
#   - HF_TOKEN env var set (export HF_TOKEN=hf_xxx)
#   - S3 bucket already created in eu-west-2
#
# Usage:
#   chmod +x transfer_setup.sh
#   S3_BUCKET=polaris-prism HF_TOKEN=hf_xxx bash transfer_setup.sh

set -euo pipefail

S3_BUCKET="${S3_BUCKET:?Set S3_BUCKET env var}"
HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var}"
LOCAL_DIR="/home/ubuntu/prism_data"
HF_REPO="NeurIPS-2026-PRISM/PRISM-Dataset"
LOG_FILE="/home/ubuntu/transfer.log"

echo "=== POLARIS: PRISM Dataset Transfer ==="
echo "S3 bucket : s3://${S3_BUCKET}/raw/"
echo "Local dir : ${LOCAL_DIR}"
echo "Log       : ${LOG_FILE}"
echo "Started   : $(date)"

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

# --- HuggingFace download ---
# snapshot_download handles resume — safe to re-run if interrupted
python3 - <<PYEOF
import os
from huggingface_hub import snapshot_download

token = os.environ["HF_TOKEN"]
local_dir = os.environ["LOCAL_DIR"]
repo_id = os.environ["HF_REPO"]

print(f"Downloading {repo_id} to {local_dir}")
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    token=token,
    resume_download=True,
    ignore_patterns=["*.md"],
)
print("Download complete.")
PYEOF

echo "HuggingFace download done: $(date)" | tee -a "${LOG_FILE}"

# --- Extract session zips ---
# HF stores each session as train/{session}.zip and val/{session}.zip
# We unzip in-place so S3 gets extracted directory trees, not zip files
echo "Extracting session zips ..."
for split in train val; do
    split_dir="${LOCAL_DIR}/${split}"
    [ -d "${split_dir}" ] || continue
    for zip_file in "${split_dir}"/*.zip; do
        [ -f "${zip_file}" ] || continue
        session_name=$(basename "${zip_file}" .zip)
        target_dir="${split_dir}/${session_name}"
        if [ ! -d "${target_dir}" ]; then
            echo "  Extracting ${zip_file} ..."
            unzip -q "${zip_file}" -d "${split_dir}"
        fi
        rm -f "${zip_file}"   # free disk space after extraction
    done
done
echo "Extraction done: $(date)" | tee -a "${LOG_FILE}"

# --- S3 sync (same region = no egress cost) ---
# --no-progress keeps logs clean; remove it if you want transfer speed shown
echo "Syncing to s3://${S3_BUCKET}/raw/ ..."
aws s3 sync "${LOCAL_DIR}" "s3://${S3_BUCKET}/raw/" \
    --region eu-west-2 \
    --no-progress \
    --storage-class STANDARD_IA \
    2>&1 | tee -a "${LOG_FILE}"

echo "S3 sync complete: $(date)" | tee -a "${LOG_FILE}"
echo "=== Transfer finished. Safe to terminate this instance. ==="

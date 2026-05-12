#!/usr/bin/env bash
# Run once on a fresh g5.2xlarge spot instance to set up the training environment
# Usage: bash training_setup.sh

set -euo pipefail

echo "=== POLARIS: Training Instance Setup ==="

# --- CUDA / driver check ---
nvidia-smi || { echo "ERROR: No GPU detected"; exit 1; }

# --- system deps ---
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip python3-venv awscli tmux htop nvme-cli

# --- NVMe local disk setup ---
# g5.2xlarge has 450GB NVMe at /dev/nvme1n1
NVME_DEV="/dev/nvme1n1"
MOUNT_POINT="/mnt/nvme"

if ! mountpoint -q "${MOUNT_POINT}"; then
    sudo mkfs.ext4 -F "${NVME_DEV}"
    sudo mkdir -p "${MOUNT_POINT}"
    sudo mount "${NVME_DEV}" "${MOUNT_POINT}"
    sudo chown ubuntu:ubuntu "${MOUNT_POINT}"
    echo "${NVME_DEV} ${MOUNT_POINT} ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab
fi

mkdir -p "${MOUNT_POINT}/polaris"
echo "NVMe mounted at ${MOUNT_POINT}, $(df -h ${MOUNT_POINT} | tail -1 | awk '{print $4}') free"

# --- Python venv ---
cd /home/ubuntu
python3 -m venv polaris
source polaris/bin/activate

pip install --quiet --upgrade pip
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install --quiet -r /home/ubuntu/POLARIS/requirements.txt

echo "Python env ready: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(0))')"

echo "=== Setup complete. Run scripts/sync_data.py to pull data from S3, then scripts/train.py ==="

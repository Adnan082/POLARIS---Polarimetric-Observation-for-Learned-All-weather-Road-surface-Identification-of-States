# POLARIS
**Polarimetric Observation for Learned All-weather Road-surface Identification of States**

Road surface state classification (dry / damp / wet / slush / snow) using the [PRISM Dataset](https://huggingface.co/datasets/NeurIPS-2026-PRISM/PRISM-Dataset).  
Demonstrates that polarimetric imaging outperforms RGB-only baselines under wet and icy conditions.

## Setup

```bash
python -m venv polaris
source polaris/bin/activate          # Windows: .\polaris\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Repo Structure

```
configs/        Training configs, model configs, session splits
src/            Source code (data, models, training, evaluation)
scripts/        Entry points (train, evaluate, precompute, sync)
aws/            EC2 setup and S3 transfer scripts
notebooks/      EDA notebook
```

## Data Pipeline

### 1. Upload PRISM to S3 (one-time, run on c5n.2xlarge in eu-west-2)
```bash
S3_BUCKET=polaris-prism-890615325560-eu-west-2-an HF_TOKEN=hf_xxx bash aws/transfer_setup.sh
```

### 2. Precompute Stokes cache (one-time)
```bash
python scripts/precompute_stokes.py --bucket polaris-prism-890615325560-eu-west-2-an --local-dir /mnt/data
```

### 3. Sync working subset to NVMe (run at start of each training job)
```bash
python scripts/sync_data.py --bucket polaris-prism-890615325560-eu-west-2-an --local-dir /mnt/nvme/polaris
```

## EDA

Run `notebooks/eda.ipynb` on 3 sample sessions before any training.  
Answers: pixel value range, DoLP signal, class balance, label format.

## Training

```bash
# RGB baseline
python scripts/train.py --config configs/train.yaml --model rgb

# Polar baseline
python scripts/train.py --config configs/train.yaml --model polar

# Fusion (main model)
python scripts/train.py --config configs/train.yaml --model fusion
```

## Evaluation

```bash
python scripts/evaluate.py --checkpoint s3://polaris-prism-890615325560-eu-west-2-an/checkpoints/checkpoint_best.pt
```

## License

Apache 2.0 — see [LICENSE](LICENSE).  
Dataset: [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (non-commercial).

"""
POLARIS training entry point.

Stages:
    python scripts/train.py --mode rgb    --data-dir /data --stokes-dir /stokes --labels-json /data/labels.json
    python scripts/train.py --mode polar  --data-dir /data --stokes-dir /stokes --labels-json /data/labels.json
    python scripts/train.py --mode fusion --data-dir /data --stokes-dir /stokes --labels-json /data/labels.json \
                            --rgb-ckpt checkpoints/best_rgb.pth --polar-ckpt checkpoints/best_polar.pth

Testing flags:
    --subset 0.1          use 10% of data (smoke test)
    --overfit-batch       overfit one batch to verify pipeline
"""

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import PRISMDataset, SURFACE_STATES
from src.data.transforms import get_spatial_transforms, get_rgb_normalize, get_polar_normalize
from src.models.fusion import build_model


# ── Helpers ──────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(args, device):
    common = dict(
        root=args.data_dir,
        labels_json=args.labels_json,
        mode=args.mode,
        use_stokes_cache=True,
        stokes_cache_dir=args.stokes_dir,
    )

    train_ds = PRISMDataset(
        split="train",
        spatial_transform=get_spatial_transforms(args.image_size, train=True),
        rgb_normalize=get_rgb_normalize(train=True),
        polar_normalize=get_polar_normalize(),
        **common,
    )
    val_ds = PRISMDataset(
        split="val",
        spatial_transform=get_spatial_transforms(args.image_size, train=False),
        rgb_normalize=get_rgb_normalize(train=False),
        polar_normalize=get_polar_normalize(),
        **common,
    )

    if args.subset < 1.0:
        n_train = max(1, int(len(train_ds) * args.subset))
        n_val   = max(1, int(len(val_ds)   * args.subset))
        idx_train = random.sample(range(len(train_ds)), n_train)
        idx_val   = random.sample(range(len(val_ds)),   n_val)
        train_ds  = Subset(train_ds, idx_train)
        val_ds    = Subset(val_ds,   idx_val)
        print(f"Subset: train={n_train}, val={n_val}")

    loader_kw = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=2 if args.workers > 0 else None,
        persistent_workers=(args.workers > 0),
    )

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)
    return train_loader, val_loader


def class_weights(dataset, num_classes: int, device) -> torch.Tensor:
    """Inverse-frequency weights to handle class imbalance."""
    counts = torch.zeros(num_classes)
    ds = dataset.dataset if isinstance(dataset, Subset) else dataset
    for s in ds.samples:
        counts[s["state"]] += 1
    counts = counts.clamp(min=1)
    weights = counts.sum() / (num_classes * counts)
    return weights.to(device)


# ── Training / validation loops ───────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, scaler, device, accum_steps):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    bar = tqdm(loader, desc="train", leave=False)
    for step, (rgb, polar, labels, _) in enumerate(bar):
        rgb    = rgb.to(device,    non_blocking=True)
        polar  = polar.to(device,  non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=(device.type == "cuda")):
            logits = model(rgb, polar)
            loss   = criterion(logits, labels) / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        bar.set_postfix(loss=f"{loss.item() * accum_steps:.4f}")

    return total_loss / len(loader)


def val_epoch(model, loader, criterion, device, num_classes):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for rgb, polar, labels, _ in tqdm(loader, desc="val  ", leave=False):
            rgb    = rgb.to(device)
            polar  = polar.to(device)
            labels = labels.to(device)
            with autocast(enabled=(device.type == "cuda")):
                logits = model(rgb, polar)
                loss   = criterion(logits, labels)
            total_loss += loss.item()
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    macro_f1  = f1_score(all_labels, all_preds, average="macro",    zero_division=0)
    per_class = f1_score(all_labels, all_preds, average=None,       zero_division=0,
                         labels=list(range(num_classes)))
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return {
        "loss":      total_loss / len(loader),
        "acc":       acc,
        "macro_f1":  macro_f1,
        "per_class": per_class.tolist(),
    }


# ── Overfit-batch test ────────────────────────────────────────────────────────

def run_overfit_test(model, loader, device, n_iters=100):
    print("\n--- Overfit batch test ---")
    model.train()
    rgb, polar, labels, _ = next(iter(loader))
    rgb, polar, labels = rgb.to(device), polar.to(device), labels.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for i in range(1, n_iters + 1):
        optimizer.zero_grad()
        loss = criterion(model(rgb, polar), labels)
        loss.backward()
        optimizer.step()
        if i % 20 == 0:
            acc = (model(rgb, polar).argmax(1) == labels).float().mean()
            print(f"  iter {i:3d}: loss={loss.item():.4f}  acc={acc.item():.0%}")

    print("Overfit test done — restart without --overfit-batch for real training.\n")
    sys.exit(0)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",         required=True, choices=["rgb", "polar", "fusion"])
    p.add_argument("--data-dir",     required=True, help="Root with train/ and val/ RGB dirs")
    p.add_argument("--stokes-dir",   required=True, help="Root with train/ and val/ .npy dirs")
    p.add_argument("--labels-json",  required=True, help="Path to PRISM labels.json")
    p.add_argument("--output-dir",   default="checkpoints")
    p.add_argument("--image-size",   type=int,   default=512)
    p.add_argument("--batch-size",   type=int,   default=16)
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--workers",      type=int,   default=4)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--accum-steps",  type=int,   default=4,    help="Gradient accumulation steps")
    p.add_argument("--subset",       type=float, default=1.0,  help="Fraction of dataset to use")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--overfit-batch",action="store_true")
    p.add_argument("--no-pretrained",action="store_true",       help="Random init (for testing)")
    # Fusion warm-start
    p.add_argument("--rgb-ckpt",     default=None, help="Best RGB checkpoint for fusion warm-start")
    p.add_argument("--polar-ckpt",   default=None, help="Best polar checkpoint for fusion warm-start")
    # S3 upload
    p.add_argument("--s3-bucket",    default=None, help="Upload best checkpoint to this S3 bucket")
    p.add_argument("--s3-prefix",    default="checkpoints")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True   # fastest conv algo for fixed 512×512 input
    set_seed(args.seed)

    print(f"Mode: {args.mode} | Device: {device} | Image: {args.image_size}px")
    print(f"Batch: {args.batch_size} x {args.accum_steps} accum = "
          f"{args.batch_size * args.accum_steps} effective")

    # ── Dataset ──────────────────────────────────────────────────────────────
    train_loader, val_loader = build_loaders(args, device)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    num_classes = len(SURFACE_STATES)

    # ── Model ─────────────────────────────────────────────────────────────────
    pretrained = not args.no_pretrained
    model = build_model(args.mode, num_classes=num_classes, pretrained=pretrained)

    if args.mode == "fusion" and args.rgb_ckpt and args.polar_ckpt:
        model.load_baseline_weights(args.rgb_ckpt, args.polar_ckpt)
        print(f"Loaded warm-start weights from {args.rgb_ckpt} + {args.polar_ckpt}")

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {total_params:.1f}M")

    if device.type == "cuda":
        model = torch.compile(model)           # fuse kernels, ~20-30% throughput gain
        print("torch.compile enabled")

    # ── Overfit test ──────────────────────────────────────────────────────────
    if args.overfit_batch:
        run_overfit_test(model, train_loader, device)

    # ── Loss, optimiser, scheduler ────────────────────────────────────────────
    weights  = class_weights(train_loader.dataset, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    print(f"Class weights: {weights.cpu().tolist()}")

    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if "head" not in n],
         "lr": args.lr},
        {"params": [p for n, p in model.named_parameters() if "head"     in n],
         "lr": args.lr * 10},
    ], weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # ── Checkpoint dir ────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_f1   = 0.0
    best_path = out_dir / f"best_{args.mode}.pth"

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, scaler, device, args.accum_steps
        )
        val_metrics = val_epoch(model, val_loader, criterion, device, num_classes)
        scheduler.step()

        is_best = val_metrics["macro_f1"] > best_f1
        if is_best:
            best_f1 = val_metrics["macro_f1"]
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_f1": best_f1,
                "args": vars(args),
            }, best_path)

        per_class_str = "  ".join(
            f"{SURFACE_STATES[i][:4]}={val_metrics['per_class'][i]:.3f}"
            for i in range(num_classes)
        )
        print(
            f"Ep {epoch:3d}/{args.epochs} | "
            f"loss {train_loss:.4f} -> {val_metrics['loss']:.4f} | "
            f"acc {val_metrics['acc']:.3f} | "
            f"F1 {val_metrics['macro_f1']:.3f} {'*' if is_best else ' '} | "
            f"[{per_class_str}] | "
            f"{time.time()-t0:.0f}s"
        )

    print(f"\nBest macro F1: {best_f1:.4f}  saved to {best_path}")

    # ── Per-class report ──────────────────────────────────────────────────────
    print("\nFinal validation report:")
    model.load_state_dict(torch.load(best_path)["model"])
    metrics = val_epoch(model, val_loader, criterion, device, num_classes)
    for i, name in enumerate(SURFACE_STATES):
        print(f"  {name:15s}  F1={metrics['per_class'][i]:.4f}")
    print(f"  {'macro':15s}  F1={metrics['macro_f1']:.4f}")

    # ── S3 upload ─────────────────────────────────────────────────────────────
    if args.s3_bucket:
        import boto3
        s3  = boto3.client("s3", region_name="eu-west-2")
        key = f"{args.s3_prefix}/{best_path.name}"
        s3.upload_file(str(best_path), args.s3_bucket, key)
        print(f"Uploaded to s3://{args.s3_bucket}/{key}")


if __name__ == "__main__":
    main()

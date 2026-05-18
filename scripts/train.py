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
import json
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
        s3_bucket=args.s3_bucket,              # enables S3 prefetch pipeline
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


# ── Plotting ──────────────────────────────────────────────────────────────────

def save_plots(history: list, out_dir: Path, mode: str, class_names: list) -> tuple:
    """Save training curves PNG + metrics JSON to out_dir."""
    import matplotlib
    matplotlib.use("Agg")          # non-interactive — no display on EC2
    import matplotlib.pyplot as plt

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=2)

    epochs   = [h["epoch"]     for h in history]
    best_idx = max(range(len(history)), key=lambda i: history[i]["macro_f1"])
    best_ep  = history[best_idx]["epoch"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"POLARIS — {mode} mode", fontsize=14, fontweight="bold")

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, [h["train_loss"] for h in history], label="train", color="steelblue")
    ax.plot(epochs, [h["val_loss"]   for h in history], label="val",   color="coral")
    ax.axvline(best_ep, color="red", linestyle="--", alpha=0.4, label=f"best ep {best_ep}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Macro F1
    ax = axes[0, 1]
    ax.plot(epochs, [h["macro_f1"] for h in history], color="mediumseagreen")
    ax.axvline(best_ep, color="red", linestyle="--", alpha=0.4, label=f"best ep {best_ep}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Macro F1"); ax.set_title("Macro F1 (val)")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Per-class F1 at best epoch
    ax = axes[1, 0]
    pc = history[best_idx]["per_class"]
    colors = ["steelblue" if v >= 0.7 else "coral" if v < 0.4 else "gold" for v in pc]
    bars = ax.bar(class_names, pc, color=colors)
    ax.set_ylim(0, 1.1); ax.set_ylabel("F1")
    ax.set_title(f"Per-class F1 (epoch {best_ep})")
    ax.tick_params(axis="x", rotation=20)
    for bar, val in zip(bars, pc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    # Val accuracy
    ax = axes[1, 1]
    ax.plot(epochs, [h["val_acc"] for h in history], color="mediumpurple")
    ax.axvline(best_ep, color="red", linestyle="--", alpha=0.4, label=f"best ep {best_ep}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.set_title("Val Accuracy")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = out_dir / "training_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plots saved to {plot_path}")
    return metrics_path, plot_path


def collect_predictions(model, loader, device) -> tuple:
    """Single val forward pass — returns (preds, labels, softmax_probs)."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for rgb, polar, labels, _ in tqdm(loader, desc="evaluating", leave=False):
            rgb, polar = rgb.to(device), polar.to(device)
            with autocast(enabled=(device.type == "cuda")):
                logits = model(rgb, polar)
            probs = torch.softmax(logits, dim=1)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.append(probs.cpu().numpy())
    return all_preds, all_labels, np.concatenate(all_probs, axis=0)


def save_confusion_matrix(all_preds, all_labels, class_names: list, out_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm      = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Confusion Matrix", fontsize=13, fontweight="bold")

    for ax, data, title in [(ax1, cm, "Counts"), (ax2, cm_norm, "Normalised")]:
        im = ax.imshow(data, cmap="Blues", vmin=0, vmax=(1.0 if data is cm_norm else None))
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                val   = data[i, j]
                text  = f"{val:.2f}" if data is cm_norm else str(int(val))
                color = "white" if (data is cm_norm and val > 0.5) else "black"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

    plt.tight_layout()
    path = out_dir / "confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def save_roc_curves(all_probs: np.ndarray, all_labels, class_names: list, out_dir: Path) -> tuple:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc, roc_auc_score

    n     = len(class_names)
    y_bin = label_binarize(all_labels, classes=list(range(n)))
    colors = plt.cm.tab10(np.linspace(0, 1, n))

    fig, ax = plt.subplots(figsize=(9, 7))

    for i, (name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name}  (AUC = {roc_auc:.3f})")

    macro_auc = roc_auc_score(y_bin, all_probs, average="macro", multi_class="ovr")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — one-vs-rest  (macro AUC = {macro_auc:.3f})")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "roc_auc.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path, macro_auc


def save_classification_report(all_preds, all_labels, class_names: list, out_dir: Path) -> Path:
    from sklearn.metrics import classification_report

    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print("\nClassification Report:\n")
    print(report)

    path = out_dir / "classification_report.txt"
    path.write_text(report)
    return path


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
    history = []

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

        history.append({
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_metrics["loss"],
            "val_acc":    val_metrics["acc"],
            "macro_f1":   val_metrics["macro_f1"],
            "per_class":  val_metrics["per_class"],
        })

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

    # ── Evaluation on best checkpoint ────────────────────────────────────────
    print("\nLoading best checkpoint for final evaluation...")
    model.load_state_dict(torch.load(best_path)["model"])

    all_preds, all_labels, all_probs = collect_predictions(model, val_loader, device)

    # Classification report (precision / recall / F1 / support per class)
    report_path = save_classification_report(all_preds, all_labels, SURFACE_STATES, out_dir)

    # Training curves (loss, macro F1, per-class F1, accuracy)
    metrics_path, curves_path = save_plots(history, out_dir, args.mode, SURFACE_STATES)

    # Confusion matrix (counts + normalised)
    cm_path = save_confusion_matrix(all_preds, all_labels, SURFACE_STATES, out_dir)

    # ROC-AUC curves (one-vs-rest, one curve per class + macro average)
    roc_path, macro_auc = save_roc_curves(all_probs, all_labels, SURFACE_STATES, out_dir)
    print(f"Macro AUC: {macro_auc:.4f}")

    # ── S3 upload — checkpoint + all artefacts ────────────────────────────────
    if args.s3_bucket:
        import boto3
        s3 = boto3.client("s3", region_name="eu-west-2")

        uploads = [
            (best_path,    f"{args.s3_prefix}/{best_path.name}"),
            (curves_path,  f"{args.s3_prefix}/{args.mode}_training_curves.png"),
            (cm_path,      f"{args.s3_prefix}/{args.mode}_confusion_matrix.png"),
            (roc_path,     f"{args.s3_prefix}/{args.mode}_roc_auc.png"),
            (report_path,  f"{args.s3_prefix}/{args.mode}_classification_report.txt"),
            (metrics_path, f"{args.s3_prefix}/{args.mode}_metrics.json"),
        ]
        for local, key in uploads:
            s3.upload_file(str(local), args.s3_bucket, key)
            print(f"Uploaded s3://{args.s3_bucket}/{key}")


if __name__ == "__main__":
    main()

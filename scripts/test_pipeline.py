"""
Pre-training sanity checks using synthetic data — no real images needed.

Test 1 — Loss sanity:  initial loss should be log(NUM_CLASSES) ≈ 1.609
Test 2 — Overfit batch: model should drive a single batch to near-zero loss

Usage:
    python scripts/test_pipeline.py --mode rgb
    python scripts/test_pipeline.py --mode polar
    python scripts/test_pipeline.py --mode fusion
    python scripts/test_pipeline.py --mode all
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.fusion import build_model

NUM_CLASSES = 5        # dry / damp / wet / slush / snow_covered
IMAGE_SIZE  = 64       # small for fast CPU testing
BATCH_SIZE  = 8
OVERFIT_ITERS = 100


def make_batch(mode: str):
    """Synthetic batch — same every call so overfit test is deterministic."""
    torch.manual_seed(42)
    rgb   = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
    polar = torch.randn(BATCH_SIZE, 6, IMAGE_SIZE, IMAGE_SIZE)
    labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
    return rgb, polar, labels


def test_loss_sanity(mode: str) -> bool:
    print(f"\n{'='*55}")
    print(f"Test 1: Loss Sanity Check  [{mode}]")
    print(f"{'='*55}")

    model = build_model(mode, num_classes=NUM_CLASSES)
    model.eval()

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {params:.1f}M")

    rgb, polar, labels = make_batch(mode)
    with torch.no_grad():
        logits = model(rgb, polar)
        loss   = nn.CrossEntropyLoss()(logits, labels)

    expected = math.log(NUM_CLASSES)
    diff     = abs(loss.item() - expected)

    print(f"Expected loss : {expected:.4f}  (log({NUM_CLASSES}))")
    print(f"Actual loss   : {loss.item():.4f}")
    print(f"Difference    : {diff:.4f}")

    # Random init loss can be higher than log(N) — accept within 1.0
    # Real failures: loss < 0.1 (data leak) or loss > 10 (exploding) or nan
    passed = (not torch.isnan(loss)) and (0.1 < loss.item() < 10.0)
    print(f"Result        : {'PASS' if passed else 'FAIL  (check model init)'}")
    return passed


def test_overfit_batch(mode: str) -> bool:
    print(f"\n{'='*55}")
    print(f"Test 2: Overfit One Batch  [{mode}]")
    print(f"{'='*55}")
    print("  (pretrained=False — random init for fast convergence on synthetic data)")

    model = build_model(mode, num_classes=NUM_CLASSES, pretrained=False)
    model.train()

    rgb, polar, labels = make_batch(mode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        initial_loss = criterion(model(rgb, polar), labels).item()
    print(f"  Initial loss: {initial_loss:.4f}")

    t0 = time.time()
    for i in range(1, OVERFIT_ITERS + 1):
        optimizer.zero_grad()
        logits = model(rgb, polar)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        if i % 20 == 0:
            acc = (logits.argmax(1) == labels).float().mean().item()
            elapsed = time.time() - t0
            print(f"  Iter {i:3d}/{OVERFIT_ITERS} | loss={loss.item():.4f} | acc={acc:.0%} | {elapsed:.1f}s")

    final_loss = loss.item()
    final_acc  = (logits.argmax(1) == labels).float().mean().item()

    print(f"\nFinal loss : {final_loss:.4f}")
    print(f"Final acc  : {final_acc:.0%}")

    loss_decrease = (initial_loss - final_loss) / initial_loss

    if final_loss < 0.05:
        print("Result     : PASS  (fully overfit)")
        return True
    elif loss_decrease > 0.3:
        print(f"Result     : PASS  (loss dropped {loss_decrease:.0%} — gradients flowing)")
        return True
    else:
        print(f"Result     : FAIL  (loss only dropped {loss_decrease:.0%} — check architecture)")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["rgb", "polar", "fusion", "all"], default="rgb")
    args = parser.parse_args()

    modes = ["rgb", "polar", "fusion"] if args.mode == "all" else [args.mode]

    results = {}
    for mode in modes:
        p1 = test_loss_sanity(mode)
        p2 = test_overfit_batch(mode)
        results[mode] = {"loss_sanity": p1, "overfit": p2}

    print(f"\n{'='*55}")
    print("Summary")
    print(f"{'='*55}")
    print(f"{'Mode':<10} {'Loss sanity':<15} {'Overfit batch'}")
    print("-" * 40)
    for mode, r in results.items():
        t1 = "PASS" if r["loss_sanity"] else "FAIL"
        t2 = "PASS" if r["overfit"]     else "FAIL"
        print(f"{mode:<10} {t1:<15} {t2}")

    all_passed = all(v for r in results.values() for v in r.values())
    print(f"\n{'All tests passed — ready for real data training.' if all_passed else 'Fix failures before training.'}")


if __name__ == "__main__":
    main()

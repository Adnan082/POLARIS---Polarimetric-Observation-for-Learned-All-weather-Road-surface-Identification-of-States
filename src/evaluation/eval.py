"""Per-condition evaluation: accuracy breakdown by weather/surface state."""

# TODO: implement after training pipeline is complete
# Will produce per-class confusion matrices and McNemar's test vs RGB baseline

from pathlib import Path
import pandas as pd
import torch


def evaluate(model, loader, condition_labels: list[str], output_csv: str):
    """
    Runs inference and writes a CSV with columns:
        frame_id, session, weather, surface, material, pred, correct
    All downstream aggregation (per-condition accuracy, confusion matrices)
    is done on this CSV — not baked into this function.
    """
    raise NotImplementedError("Implement after DataLoader is ready")

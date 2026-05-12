"""Main training entry point. Implement after DataLoader is complete."""

# TODO: wire up after EDA and DataLoader implementation

import argparse
import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--model", choices=["rgb", "polar", "fusion"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("Training not yet implemented — complete DataLoader first.")

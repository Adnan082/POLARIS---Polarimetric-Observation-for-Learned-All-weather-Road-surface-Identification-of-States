"""
PRISM dataset loader.

Layout inside each session ZIP (after extraction):
    {session}/
        sequence_001/
            rgb/                *.png  (seconds_ms timestamp stems e.g. 1769664808_810)
            polar/
                0d/             *.png
                45d/            *.png
                90d/            *.png
                135d/           *.png
            lidar_accum_scan/   *.pcd
        sequence_002/
            ...
        vehicle_state/          (session-level)

Images: uint8 PNG (0-255). Polar resolution is half of RGB (2x2 super-pixel sensor).

Label lookup (3-step, from labels.json metadata):
    1. timestamp_overrides  — if frame ts falls in any override window, use it
    2. ordered_segments     — find segment by ts_start/ts_end boundaries
    3. default_label        — fallback for homogeneous sessions
"""

from pathlib import Path
from typing import Literal
import json

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.stokes import compute_stokes, pack_polar_channels
from src.data.utils import (
    POLAR_DIRS, lookup_label, _parse_frame_ts, _parse_ts, imread_unicode
)

# keep aliases for backward compat
_parse_label_ts = _parse_ts
_ns_to_sec      = _parse_frame_ts

SURFACE_STATES    = ["dry", "damp", "wet", "slush", "snow_covered"]
SURFACE_MATERIALS = ["asphalt", "concrete", "belgian_block", "gravel", "other"]
WEATHER           = ["clear", "overcast", "rainy", "foggy", "snowy"]


class PRISMDataset(Dataset):
    def __init__(
        self,
        root: str,
        labels_json: str,
        split: Literal["train", "val"],
        mode: Literal["rgb", "polar", "fusion"],
        spatial_transform=None,
        rgb_normalize=None,
        polar_normalize=None,
        use_stokes_cache: bool = True,
        stokes_cache_dir: str | None = None,
    ):
        self.root             = Path(root)
        self.mode             = mode
        self.spatial_transform = spatial_transform
        self.rgb_normalize    = rgb_normalize
        self.polar_normalize  = polar_normalize
        self.use_stokes_cache = use_stokes_cache
        self.stokes_cache_dir = Path(stokes_cache_dir) if stokes_cache_dir else None

        with open(labels_json) as f:
            raw = json.load(f)
        self.folders = raw["folders"]

        self.samples = self._index(split)
        print(f"PRISMDataset [{split}]: {len(self.samples)} frames, mode={mode}")

    def _index(self, split: str) -> list[dict]:
        split_root = self.root / split
        samples    = []

        for session_dir in sorted(split_root.iterdir()):
            if not session_dir.is_dir():
                continue
            session_id   = session_dir.name
            folder_entry = self.folders.get(session_id)
            if folder_entry is None:
                continue

            for seq_dir in sorted(session_dir.iterdir()):
                if not seq_dir.is_dir() or seq_dir.name == "vehicle_state":
                    continue
                rgb_dir = seq_dir / "rgb"
                if not rgb_dir.exists():
                    continue

                for rgb_path in sorted(rgb_dir.glob("*.png")):
                    stem   = rgb_path.stem
                    ts_sec = _parse_frame_ts(stem)
                    label  = lookup_label(folder_entry, ts_sec)

                    if label is None:
                        continue

                    state_idx    = SURFACE_STATES.index(label["surface_state"]) \
                                   if label["surface_state"] in SURFACE_STATES else -1
                    material_idx = SURFACE_MATERIALS.index(label["surface_material"]) \
                                   if label["surface_material"] in SURFACE_MATERIALS else -1

                    if state_idx == -1 or material_idx == -1:
                        continue

                    stokes_cache_path = None
                    if self.stokes_cache_dir:
                        stokes_cache_path = str(
                            self.stokes_cache_dir / split / session_id / seq_dir.name / f"{stem}.npy"
                        )

                    samples.append({
                        "session":          session_id,
                        "sequence":         seq_dir.name,
                        "stem":             stem,
                        "rgb_path":         str(rgb_path),
                        "polar_root":       str(seq_dir / "polar"),
                        "stokes_cache_path": stokes_cache_path,
                        "state":            state_idx,
                        "material":         material_idx,
                        "weather":          label.get("weather", ""),
                        "road_type":        label.get("road_type", ""),
                    })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        need_rgb   = self.mode in ("rgb",   "fusion")
        need_polar = self.mode in ("polar", "fusion")

        rgb_img    = self._load_rgb(s["rgb_path"])   if need_rgb   else None
        polar_data = self._load_polar(
            s["polar_root"], s["stem"], s.get("stokes_cache_path")
        ) if need_polar else None

        # Apply same spatial transform to both streams (same crop, same flip)
        if self.spatial_transform is not None:
            inp = {}
            if rgb_img   is not None: inp["image"] = rgb_img
            if polar_data is not None: inp["polar"] = polar_data.transpose(1, 2, 0)
            if not inp:
                raise RuntimeError("No inputs for spatial transform")
            if "image" not in inp:
                inp["image"] = inp["polar"]   # albumentations needs "image" key
                result = self.spatial_transform(**inp)
                polar_hwc = result["image"]
            elif "polar" not in inp:
                result = self.spatial_transform(**inp)
                rgb_img = result["image"]
                polar_hwc = None
            else:
                result = self.spatial_transform(**inp)
                rgb_img   = result["image"]
                polar_hwc = result["polar"]
        else:
            polar_hwc = polar_data.transpose(1, 2, 0) if polar_data is not None else None

        # Stream-specific normalisation + ToTensor
        if rgb_img is not None and self.rgb_normalize is not None:
            rgb_img = self.rgb_normalize(image=rgb_img)["image"]
        if polar_hwc is not None and self.polar_normalize is not None:
            polar_data = self.polar_normalize(image=polar_hwc)["image"]

        # Dummy tensors for unused stream so DataLoader can collate
        if rgb_img   is None: rgb_img   = torch.zeros(3, 1, 1)
        if polar_data is None: polar_data = torch.zeros(6, 1, 1)

        label = torch.tensor(s["state"], dtype=torch.long)

        return rgb_img, polar_data, label, {
            "session":   s["session"],
            "weather":   s["weather"],
            "road_type": s["road_type"],
            "material":  s["material"],
        }

    def _load_rgb(self, path: str) -> np.ndarray:
        import cv2
        img = imread_unicode(path)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32) / 255.0

    def _load_polar(self, polar_root: str, stem: str, cache_path: str | None = None) -> np.ndarray:
        if self.use_stokes_cache and cache_path:
            p = Path(cache_path)
            if p.exists():
                return np.load(str(p))

        root   = Path(polar_root)
        arrays = {}
        for angle, subdir in POLAR_DIRS.items():
            p   = root / subdir / f"{stem}.png"
            arr = imread_unicode(str(p))
            if arr is None:
                raise FileNotFoundError(str(p))
            arrays[angle] = arr.astype(np.float32)

        stokes = compute_stokes(arrays["0"], arrays["45"], arrays["90"], arrays["135"])
        return pack_polar_channels(stokes)

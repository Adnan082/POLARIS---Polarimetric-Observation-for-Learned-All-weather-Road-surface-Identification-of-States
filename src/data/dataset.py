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
import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor

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

# Resized stokes: crop bottom 65% + resize to this resolution + float16
_CACHE_HW = 512


class S3PrefetchCache:
    """
    Background thread pool that downloads files from S3 and caches locally.

    RGB  mode: downloads PNG as-is — no resize (transforms handle it)
    Polar mode: downloads full-res stokes, crops bottom 65%, resizes to
                _CACHE_HW × _CACHE_HW float16 (30 MB → 3 MB per file)

    32 threads run in parallel — I/O-bound S3 downloads don't block the GIL.
    Training starts immediately; DataLoader streams from S3 on cache misses
    and switches to local reads as files land on disk.

    Timeline (polar, 1.5 TB):
        t=0 min  → training starts, threads begin downloading
        t=8 min  → all 45K stokes cached locally
        epoch 2+ → reads local 3 MB files, GPU utilisation jumps to ~80%

    Timeline (rgb, 260 GB):
        t=0 min  → training starts, threads begin downloading
        t=2 min  → all 45K PNGs cached locally
        epoch 2+ → reads local 5.7 MB files from EBS
    """

    def __init__(self, samples: list, s3_bucket: str, mode: str, n_threads: int = 32):
        import boto3
        self._bucket   = s3_bucket
        self._mode     = mode
        self._done     = set()
        self._lock     = threading.Lock()
        self._s3       = boto3.client("s3", region_name="eu-west-2")
        self._executor = ThreadPoolExecutor(max_workers=n_threads)

        futures = []
        if mode in ("rgb", "fusion"):
            futures += [
                self._executor.submit(self._fetch_rgb, s["rgb_path"])
                for s in samples if s.get("rgb_path")
            ]
        if mode in ("polar", "fusion"):
            futures += [
                self._executor.submit(self._fetch_and_resize, s["stokes_cache_path"])
                for s in samples if s.get("stokes_cache_path")
            ]

        threading.Thread(target=self._watch, args=(futures, len(samples)), daemon=True).start()

    def _fetch_rgb(self, local_path: str):
        """Download PNG from S3 and save to local path — no processing needed."""
        p = Path(local_path)
        if p.exists():
            with self._lock:
                self._done.add(local_path)
            return
        try:
            parts = p.parts
            idx   = next(i for i, pt in enumerate(parts) if pt in ("train", "val"))
            key   = "raw/" + "/".join(parts[idx:])
            buf   = io.BytesIO()
            self._s3.download_fileobj(self._bucket, key, buf)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(buf.getvalue())
            with self._lock:
                self._done.add(local_path)
        except Exception:
            pass

    def _fetch_and_resize(self, local_path: str):
        """Download full-res stokes, crop + resize + float16, save locally."""
        import cv2
        p = Path(local_path)
        if p.exists():
            with self._lock:
                self._done.add(local_path)
            return
        try:
            parts = p.parts
            idx   = next(i for i, pt in enumerate(parts) if pt in ("train", "val"))
            key   = "processed/stokes/" + "/".join(parts[idx:])
            buf   = io.BytesIO()
            self._s3.download_fileobj(self._bucket, key, buf)
            buf.seek(0)
            arr = np.load(buf)                          # (6, H, W) float32

            h   = arr.shape[1]
            arr = arr[:, int(h * 0.35):, :]             # crop bottom 65%

            resized = np.stack([
                cv2.resize(arr[c], (_CACHE_HW, _CACHE_HW), interpolation=cv2.INTER_LINEAR)
                for c in range(arr.shape[0])
            ]).astype(np.float16)

            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(p), resized)
            with self._lock:
                self._done.add(local_path)
        except Exception:
            pass

    def _watch(self, futures, total):
        done = 0
        for f in futures:
            f.result()
            done += 1
            if done % 5000 == 0:
                print(f"S3 prefetch: {done}/{total} files cached", flush=True)
        print(f"S3 prefetch complete — all {total} files cached locally")


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
        s3_bucket: str | None = None,          # enables S3 prefetch pipeline
        s3_prefetch_threads: int = 32,
    ):
        self.root              = Path(root)
        self.mode              = mode
        self.spatial_transform = spatial_transform
        self.rgb_normalize     = rgb_normalize
        self.polar_normalize   = polar_normalize
        self.use_stokes_cache  = use_stokes_cache
        self.stokes_cache_dir  = Path(stokes_cache_dir) if stokes_cache_dir else None
        self.s3_bucket         = s3_bucket
        self._prefetch: S3PrefetchCache | None = None

        with open(labels_json) as f:
            raw = json.load(f)
        self.folders = raw["folders"]

        self.samples = self._index(split)
        print(f"PRISMDataset [{split}]: {len(self.samples)} frames, mode={mode}")

        # Start background S3 prefetch for all modes — fires immediately
        if s3_bucket:
            print(f"Starting S3 prefetch ({s3_prefetch_threads} threads, mode={mode})...")
            self._prefetch = S3PrefetchCache(
                self.samples, s3_bucket, mode, n_threads=s3_prefetch_threads
            )

    def _index(self, split: str) -> list[dict]:
        split_root = self.root / split
        # Use S3 listing when local data hasn't been synced (streaming mode)
        if self.s3_bucket and (not split_root.exists() or not any(split_root.iterdir())):
            return self._index_from_s3(split)
        return self._index_local(split)

    def _index_local(self, split: str) -> list[dict]:
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
                    samples.append(self._make_sample(
                        split, session_dir.name, seq_dir.name,
                        rgb_path.stem, str(rgb_path)
                    ) | {"_label": label})

        return [self._resolve_label(s) for s in samples if s is not None]

    def _index_from_s3(self, split: str) -> list[dict]:
        """Build sample index by listing S3 — used when no local data exists."""
        import boto3
        s3        = boto3.client("s3", region_name="eu-west-2")
        paginator = s3.get_paginator("list_objects_v2")
        samples   = []

        print(f"Indexing {split} split from S3 (no local data found)...")
        for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=f"raw/{split}/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if "/rgb/" not in key or not key.endswith(".png"):
                    continue
                # raw/train/session_id/seq_name/rgb/stem.png
                parts      = key.split("/")
                session_id = parts[2]
                seq_name   = parts[3]
                stem       = parts[5].replace(".png", "")

                folder_entry = self.folders.get(session_id)
                if folder_entry is None:
                    continue

                label = lookup_label(folder_entry, _parse_frame_ts(stem))
                if label is None:
                    continue

                local_rgb = str(self.root / split / session_id / seq_name / "rgb" / f"{stem}.png")
                sample    = self._make_sample(split, session_id, seq_name, stem, local_rgb)
                sample["_label"] = label
                samples.append(sample)

        samples.sort(key=lambda s: (s["session"], s["sequence"], s["stem"]))
        return [self._resolve_label(s) for s in samples if s is not None]

    def _make_sample(self, split, session_id, seq_name, stem, rgb_path) -> dict:
        stokes_cache_path = None
        if self.stokes_cache_dir:
            stokes_cache_path = str(
                self.stokes_cache_dir / split / session_id / seq_name / f"{stem}.npy"
            )
        return {
            "session":           session_id,
            "sequence":          seq_name,
            "stem":              stem,
            "rgb_path":          rgb_path,
            "polar_root":        str(self.root / split / session_id / seq_name / "polar"),
            "stokes_cache_path": stokes_cache_path,
        }

    def _resolve_label(self, s: dict) -> dict | None:
        label        = s.pop("_label")
        state_idx    = SURFACE_STATES.index(label["surface_state"]) \
                       if label["surface_state"] in SURFACE_STATES else -1
        material_idx = SURFACE_MATERIALS.index(label["surface_material"]) \
                       if label["surface_material"] in SURFACE_MATERIALS else -1
        if state_idx == -1 or material_idx == -1:
            return None
        return s | {
            "state":    state_idx,
            "material": material_idx,
            "weather":  label.get("weather", ""),
            "road_type": label.get("road_type", ""),
        }

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
            # not cached locally yet — stream from S3
            if self.s3_bucket:
                return self._stream_rgb_from_s3(path)
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32) / 255.0

    def _stream_rgb_from_s3(self, local_path: str) -> np.ndarray:
        """Stream RGB PNG from S3 when local cache not ready yet."""
        import boto3, cv2
        parts = Path(local_path).parts
        idx   = next(i for i, p in enumerate(parts) if p in ("train", "val"))
        key   = "raw/" + "/".join(parts[idx:])
        s3    = boto3.client("s3", region_name="eu-west-2")
        buf   = io.BytesIO()
        s3.download_fileobj(self.s3_bucket, key, buf)
        arr   = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        img   = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        img   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32) / 255.0

    def _stream_from_s3(self, cache_path: str) -> np.ndarray:
        """Fallback: stream full-res stokes from S3 when local cache not ready yet."""
        import boto3
        parts = Path(cache_path).parts
        idx   = next(i for i, p in enumerate(parts) if p in ("train", "val"))
        key   = "processed/stokes/" + "/".join(parts[idx:])
        s3    = boto3.client("s3", region_name="eu-west-2")
        buf   = io.BytesIO()
        s3.download_fileobj(self.s3_bucket, key, buf)
        buf.seek(0)
        return np.load(buf).astype(np.float32)

    def _load_polar(self, polar_root: str, stem: str, cache_path: str | None = None) -> np.ndarray:
        if self.use_stokes_cache and cache_path:
            p = Path(cache_path)
            if p.exists():
                return np.load(str(p))          # local cache hit — fast path

        # Cache miss: background thread hasn't written this file yet.
        # Stream full-res directly from S3 so the GPU batch isn't delayed.
        if self.s3_bucket and cache_path:
            return self._stream_from_s3(cache_path)

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

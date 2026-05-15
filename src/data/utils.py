"""
Pure-Python utilities — no torch dependency.
Safe to import even if torch is not installed.
"""

from pathlib import Path
import cv2
import numpy as np

POLAR_DIRS = {"0": "0d", "45": "45d", "90": "90d", "135": "135d"}


def _parse_ts(stem: str) -> float:
    """'1769664808_810' → 1769664808.810 seconds."""
    parts = stem.split("_")
    return float(parts[0]) + float(parts[1]) / 1000.0

# aliases
_parse_frame_ts = _parse_ts
_parse_label_ts = _parse_ts


def lookup_label(folder_entry: dict, frame_ts_sec: float) -> dict | None:
    """
    3-step PRISM label lookup for a single frame timestamp.
    Returns label dict or None if frame is in a transition zone.
    """
    for override in folder_entry.get("timestamp_overrides", []):
        ts_start = _parse_ts(override["ts_start"])
        ts_end   = _parse_ts(override["ts_end"])
        if ts_start <= frame_ts_sec <= ts_end:
            return override["label"]

    segments = folder_entry.get("ordered_segments", [])
    if segments:
        for seg in segments:
            ts_start = _parse_ts(seg["ts_start"])
            ts_end   = _parse_ts(seg["ts_end"])
            if ts_start <= frame_ts_sec <= ts_end:
                return seg["label"]
        return None

    if "default_label" in folder_entry:
        return folder_entry["default_label"]

    return None


def imread_unicode(path: str | Path, flags=cv2.IMREAD_UNCHANGED) -> np.ndarray | None:
    """
    cv2.imread replacement that handles Unicode paths on Windows.
    cv2.imread uses the C ANSI API and silently fails on paths with
    non-ASCII characters (e.g. em-dash in folder names).
    """
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(buf, flags)
    except Exception:
        return None

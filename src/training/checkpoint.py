"""S3-backed checkpointing with spot-safe resume logic."""

import io
import logging
from pathlib import Path

import boto3
import torch

logger = logging.getLogger(__name__)


class S3Checkpointer:
    def __init__(self, bucket: str, prefix: str, local_dir: str):
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.s3 = boto3.client("s3")

    def save(self, state: dict, epoch: int, is_best: bool = False):
        local_path = self.local_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(state, local_path)

        # always overwrite latest for fast resume after spot reclaim
        self._upload(local_path, f"{self.prefix}/checkpoint_latest.pt")
        self._upload(local_path, f"{self.prefix}/checkpoint_epoch_{epoch:03d}.pt")

        if is_best:
            self._upload(local_path, f"{self.prefix}/checkpoint_best.pt")

        logger.info("Checkpoint saved: epoch %d → s3://%s/%s", epoch, self.bucket, self.prefix)

    def load_latest(self) -> dict | None:
        key = f"{self.prefix}/checkpoint_latest.pt"
        local_path = self.local_dir / "checkpoint_latest.pt"
        try:
            self.s3.download_file(self.bucket, key, str(local_path))
            state = torch.load(local_path, map_location="cpu")
            logger.info("Resumed from s3://%s/%s (epoch %d)", self.bucket, key, state.get("epoch", -1))
            return state
        except self.s3.exceptions.ClientError:
            logger.info("No checkpoint found at s3://%s/%s — starting fresh", self.bucket, key)
            return None

    def _upload(self, local_path: Path, s3_key: str):
        self.s3.upload_file(str(local_path), self.bucket, s3_key)

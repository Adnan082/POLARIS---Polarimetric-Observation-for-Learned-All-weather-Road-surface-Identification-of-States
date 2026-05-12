"""
Polls EC2 spot instance termination metadata endpoint every 20 seconds.
When a termination notice is detected (2-minute warning), fires a callback
so the trainer can save a checkpoint and exit cleanly.

Usage:
    handler = SpotTerminationHandler(on_terminate=trainer.emergency_checkpoint)
    handler.start()   # runs in background thread
"""

import threading
import time
import urllib.request
import urllib.error
import logging

logger = logging.getLogger(__name__)

METADATA_URL = "http://169.254.169.254/latest/meta-data/spot/termination-time"
POLL_INTERVAL = 20


class SpotTerminationHandler:
    def __init__(self, on_terminate):
        self._on_terminate = on_terminate
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def start(self):
        self._thread.start()
        logger.info("Spot termination handler started (polling every %ds)", POLL_INTERVAL)

    def stop(self):
        self._stop_event.set()

    def _poll(self):
        while not self._stop_event.is_set():
            if self._termination_imminent():
                logger.warning("SPOT TERMINATION NOTICE received — saving checkpoint")
                try:
                    self._on_terminate()
                except Exception as e:
                    logger.error("Emergency checkpoint failed: %s", e)
                break
            self._stop_event.wait(POLL_INTERVAL)

    def _termination_imminent(self) -> bool:
        try:
            req = urllib.request.Request(METADATA_URL)
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status == 200
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False          # no notice yet
            return False
        except Exception:
            return False              # network hiccup — don't false-positive

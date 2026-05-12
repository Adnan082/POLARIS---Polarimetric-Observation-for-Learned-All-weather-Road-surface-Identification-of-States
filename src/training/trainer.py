"""Training loop: AMP + gradient accumulation + S3 checkpointing."""

# TODO: implement after DataLoader is complete (post-EDA)
# Skeleton only — real implementation in Phase 4

import logging
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, optimizer, scheduler, checkpointer, config, spot_handler=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpointer = checkpointer
        self.cfg = config
        self.spot_handler = spot_handler
        self.scaler = GradScaler()
        self.current_epoch = 0

    def fit(self, train_loader, val_loader):
        state = self.checkpointer.load_latest()
        if state:
            self.model.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.current_epoch = state["epoch"] + 1

        if self.spot_handler:
            self.spot_handler.start()

        for epoch in range(self.current_epoch, self.cfg.training.epochs):
            self.current_epoch = epoch
            train_loss = self._train_epoch(train_loader)
            val_metrics = self._val_epoch(val_loader)

            self.checkpointer.save(
                {"epoch": epoch, "model": self.model.state_dict(),
                 "optimizer": self.optimizer.state_dict()},
                epoch=epoch,
                is_best=val_metrics.get("is_best", False),
            )
            self.scheduler.step()
            logger.info("Epoch %d | loss %.4f | val_acc %.4f", epoch, train_loss, val_metrics["acc"])

    def emergency_checkpoint(self):
        """Called by SpotTerminationHandler — saves immediately."""
        self.checkpointer.save(
            {"epoch": self.current_epoch, "model": self.model.state_dict(),
             "optimizer": self.optimizer.state_dict(), "emergency": True},
            epoch=self.current_epoch,
        )
        logger.warning("Emergency checkpoint saved at epoch %d", self.current_epoch)

    def _train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        accum_steps = self.cfg.training.grad_accumulation_steps
        self.optimizer.zero_grad()

        for step, batch in enumerate(loader):
            rgb, polar, labels = batch
            rgb = rgb.cuda(non_blocking=True)
            polar = polar.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            with autocast():
                logits = self.model(rgb, polar)
                loss = nn.CrossEntropyLoss()(logits, labels) / accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * accum_steps

        return total_loss / len(loader)

    def _val_epoch(self, loader):
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in loader:
                rgb, polar, labels = batch
                rgb, polar, labels = rgb.cuda(), polar.cuda(), labels.cuda()
                with autocast():
                    logits = self.model(rgb, polar)
                correct += (logits.argmax(1) == labels).sum().item()
                total += labels.size(0)
        return {"acc": correct / total}

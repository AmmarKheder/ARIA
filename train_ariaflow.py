#!/usr/bin/env python3
"""
ARIA-Flow Training Script — LUMI (16 nodes × 8 MI250X = 128 GPUs).
Uses existing GlobalARIADataModule (same dataset, same batch format).
"""
import os
import sys
import argparse
from pathlib import Path

import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ariaflow.trainer_flow import ARIAFlowLightning
from dataset_global import GlobalARIADataModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/scratch/project_462001140/ammar/eccv/aria/configs/ariaflow_pretrain.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    pl.seed_everything(42, workers=True)

    datamodule = GlobalARIADataModule(config)
    model = ARIAFlowLightning(config)

    tc = config["train"]
    checkpoint_cb = ModelCheckpoint(
        dirpath=f"/scratch/project_462001140/ammar/eccv/aria/checkpoints_ariaflow",
        filename="ariaflow-{epoch:03d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=tc.get("epochs", 300),
        precision=tc.get("precision", "bf16-mixed"),
        strategy="ddp_find_unused_parameters_false",
        devices=8,
        num_nodes=int(os.environ.get("SLURM_NNODES", 1)),
        gradient_clip_val=tc.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=tc.get("accumulate_grad_batches", 1),
        log_every_n_steps=tc.get("log_every_n_steps", 10),
        check_val_every_n_epoch=tc.get("check_val_every_n_epoch", 1),
        callbacks=[checkpoint_cb, LearningRateMonitor("step")],
        logger=TensorBoardLogger(
            save_dir="/scratch/project_462001140/ammar/eccv/aria/logs_ariaflow",
            name="ariaflow",
        ),
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume)


if __name__ == "__main__":
    main()

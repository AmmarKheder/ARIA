#!/usr/bin/env python3
"""
ARIA Global Training Script.
Reuses the Europe model architecture unchanged, only the dataset differs.
"""
import os, sys
import argparse
from pathlib import Path

import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# Use the ARIA model and trainer from the site package
CRANPM_PATH = Path("/scratch/project_462001140/ammar/eccv/topoflow_europe/aria_site")
sys.path.insert(0, str(CRANPM_PATH))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aria.training.trainer import ARIALightning
from dataset_global import GlobalARIADataModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/scratch/project_462001140/ammar/eccv/aria/configs/global_pretrain.yaml")
    parser.add_argument("--resume",   default=None)
    parser.add_argument("--finetune", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    pl.seed_everything(42, workers=True)

    datamodule = GlobalARIADataModule(config)
    model = ARIALightning(config)

    if args.finetune:
        ckpt = torch.load(args.finetune, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"Loaded finetune weights from {args.finetune}")

    ckpt_dir = os.environ.get("CKPT_DIR", "/scratch/project_462001140/ammar/eccv/aria/checkpoints_global")
    log_dir  = os.environ.get("LOG_DIR",  "/scratch/project_462001140/ammar/eccv/aria/logs_global")

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="aria_global-{epoch:03d}-val{val/rmse:.3f}",
            monitor="val/rmse", mode="min",
            save_top_k=config["train"].get("save_top_k", 3),
            save_last=True, auto_insert_metric_name=False,
        ),
        EarlyStopping(monitor="val/rmse", patience=config["train"].get("early_stopping_patience", 100), mode="min"),
        LearningRateMonitor(logging_interval="step"),
    ]

    logger = TensorBoardLogger(save_dir=log_dir, name="aria_global", default_hp_metric=False)

    tc = config["train"]
    trainer = pl.Trainer(
        max_epochs=tc["epochs"],
        precision=tc.get("precision", "bf16-mixed"),
        gradient_clip_val=tc.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=tc.get("accumulate_grad_batches", 4),
        log_every_n_steps=tc.get("log_every_n_steps", 10),
        check_val_every_n_epoch=tc.get("check_val_every_n_epoch", 1),
        callbacks=callbacks,
        logger=logger,
        strategy="ddp_find_unused_parameters_false",
        devices=-1,
        num_nodes=int(os.environ.get("SLURM_NNODES", 1)),
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume)
    print("Training complete.")


if __name__ == "__main__":
    main()

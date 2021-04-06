#!/usr/bin/env python3
# -*- coding=utf-8 -*-
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from DataClasses import KGQGDataset, KGQGDataModule
from KGQGTuner import KGQGTuner


if __name__ == '__main__':
    kgqg = KGQGDataModule('data/WQ')
    model = KGQGTuner(kgqg)

    # Autoscale batch size
    trainer = pl.Trainer(auto_scale_batch_size='power', auto_lr_find=True,
                        max_epochs=3, progress_bar_refresh_rate=20, callbacks=[EarlyStopping(monitor='val_loss')])

    # Find the batch size en lr
    trainer.tune(model, datamodule=kgqg)

    # Train model
    # trainer.fit(model)

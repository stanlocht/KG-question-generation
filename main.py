#!/usr/bin/env python3
# -*- coding=utf-8 -*-
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers

from DataClasses import KGQGDataset, KGQGDataModule
from KGQGTuner import KGQGTuner


if __name__ == "__main__":
    # set random seed
    pl.seed_everything(42)

    parser = ArgumentParser()
    
    # add model specific args
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--dataset', type=str, default='WQ')
    parser.add_argument('--logdir', type=str, default='logs')
    parser.add_argument('--pre_trained', type=str, default='t5', help='t5 or bart')

    # add all the available trainer options to argparse
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Define trainer
    tb_logger = pl_loggers.TensorBoardLogger(args.logdir+'/')
    trainer = pl.Trainer.from_argparse_args( 
        args,  # max_epochs, gpus
        logger=tb_logger,
        callbacks=[EarlyStopping(monitor='bleu_score', verbose=True, mode='max', patience=5)]
        )

    # Load data and model
    kgqg = KGQGDataModule('data/' + args.dataset, batch_size=args.batch_size, pre_trained=args.pre_trained)
    model = KGQGTuner(kgqg, learning_rate=args.learning_rate, batch_size=args.batch_size,
                      optimizer=args.optimizer,dataset=args.dataset, pre_trained=args.pre_trained)

    # Fit model
    trainer.fit(model, datamodule=kgqg)
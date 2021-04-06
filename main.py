#!/usr/bin/env python3
# -*- coding=utf-8 -*-
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers

from DataClasses import KGQGDataset, KGQGDataModule
from KGQGTuner import KGQGTuner

def write_test_files(model, datamodule, name='', prettyfile=True):
    out_dir = Path.cwd() / 'test_results' / name
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    with open(out_dir/'predictions.txt', 'w', encoding='utf-8') as f:
        [f.write(e + '\n') for e in model.test_preds]

    with open(out_dir/'targets.txt', 'w', encoding='utf-8') as f:
        [f.write(e + '\n') for e in model.test_targets]
      
    if prettyfile:
        with open(out_dir/'predsandtgts.txt', 'w', encoding='utf-8') as f:
            for i in range(len(model.test_preds)):
                f.write(f'===============Datapoint {i}======================\n')
                f.write('Source graph:\n')
                f.write(f'{datamodule.test_set.source[i]}\n')
                f.write('Target question:\n')
                f.write(f'{model.test_targets[i]}\n')
                f.write('Predicted question:\n')
                f.write(f'{model.test_preds[i]}\n\n')


if __name__ == "__main__":
    parser = ArgumentParser()
    
    # add model specific args
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--dataset', type=str, default='WQ')

    # add all the available trainer options to argparse
    parser = pl.Trainer.add_argparse_args(parser)
    
    parser.add_argument("--test", help="Test model after fit.", 
                        action="store_true")
    parser.add_argument("--savename", type=str, default='no_name')

    args = parser.parse_args()

    # Define trainer
    tb_logger = pl_loggers.TensorBoardLogger('logs/')
    trainer = pl.Trainer.from_argparse_args( 
        args,  # max_epochs, gpus
        logger=tb_logger,
        callbacks=[EarlyStopping(monitor='val_loss')]
        )

    # Load data and model
    kgqg = KGQGDataModule('data/' + args.dataset)
    model = KGQGTuner(kgqg, args.learning_rate, args.batch_size, args.dataset)

    # Fit model
    trainer.fit(model, datamodule=kgqg)

    # Test model
    if args.test:
        trainer.test(model=model, datamodule=kgqg)
        write_test_files(model, kgqg, name=args.savename)
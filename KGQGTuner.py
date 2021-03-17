#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    )
import pytorch_lightning as pl

class KGQGTuner(pl.LightningModule):
    def __init__(self, datamodule, learning_rate=3e-5):
        super(KGQGTuner, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained('t5-base')
        # resize embedding to account for additional special tokens
        self.tokenizer = datamodule.tokenizer
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.learning_rate = learning_rate
        
        # add batch size to init to enable automatic batch size scaling.
        self.batch_size = datamodule.batch_size
        
        
    def forward(
          self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None
        ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids, # Can be None when passing labels
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        # make sure no loss is calculated on the pad tokens by setting these to -100
        labels = batch["target_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=labels,
            decoder_attention_mask=batch['target_mask']
        )
        return outputs.loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('val_loss', loss)
    
    def configure_optimizers(self, eps=1e-8):
        return AdamW(self.model.parameters(), lr=self.learning_rate, eps=eps)
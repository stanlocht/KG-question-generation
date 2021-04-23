#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from transformers import (
    AdamW,
    Adafactor,
    T5ForConditionalGeneration,
    BartForConditionalGeneration
    )
import pytorch_lightning as pl
from torchtext.data.metrics import bleu_score

class KGQGTuner(pl.LightningModule):
    def __init__(self, datamodule, learning_rate=3e-5, batch_size=8, optimizer='adam', dataset='', pre_trained='t5'):
        super(KGQGTuner, self).__init__()

        if pre_trained:
            self.model = T5ForConditionalGeneration.from_pretrained('t5-base')
        else:
            self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        
        # resize embedding to account for additional special tokens
        self.tokenizer = datamodule.tokenizer
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.learning_rate = learning_rate
        
        # add batch size to init to enable automatic batch size scaling.
        self.batch_size = datamodule.batch_size
        #self.dataset = dataset
        self.optimizer = optimizer

        # testing
        self.bleu_metric = bleu_score
        
        self.save_hyperparameters('learning_rate', 'batch_size','optimizer','dataset', 'pre_trained')

        
    def forward(
          self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None
        ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids, # can be left None when passing labels
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
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Generate predictions
        preds_tokens = self.model.generate(batch['source_ids'])
        preds = self.tokenizer.batch_decode(preds_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        targets = self.tokenizer.batch_decode(batch['target_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        # Calculate loss
        loss = self._step(batch)
        self.log('val_loss', loss, sync_dist=True)
        return {'preds' : preds, 'targets' : targets}
    
    def validation_epoch_end(self, outputs):
        # Get list of all preds and targets of the epoch
        self.val_preds = [pred for output in outputs for pred in output['preds']]
        self.val_targets = [target for output in outputs for target in output['targets']]
        
        # Calculate BLEU score (max n=4)
        splitpreds = [pred.split() for pred in self.val_preds]
        splittargets = [[target.split()] for target in self.val_targets]
        self.bleu = self.bleu_metric(splitpreds, splittargets)
        self.log('bleu_score', self.bleu)
        
    def test_step(self, batch, batch_idx):       
        # Generate predictions
        preds_tokens = self.model.generate(batch['source_ids'], max_length=50)
        preds = self.tokenizer.batch_decode(preds_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        targets = self.tokenizer.batch_decode(batch['target_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        # Calculate loss
        loss = self._step(batch)
        self.log('test_loss', loss, sync_dist=True)
        return {'preds' : preds, 'targets' : targets}
    
        
    def test_epoch_end(self, outputs):
        print(outputs)
        # Get list of all preds and targets of the epoch
        self.test_preds = [pred for output in outputs for pred in output['preds']]
        self.test_targets = [target for output in outputs for target in output['targets']]
        
        # Calculate BLEU score (max n=4)
        splitpreds = [pred.split() for pred in self.test_preds]
        splittargets = [[target.split()] for target in self.test_targets]
        bleu = self.bleu_metric(splitpreds, splittargets)
        self.log('bleu_score', bleu)
        
    
    def configure_optimizers(self, eps=1e-8):
        if self.optimizer == 'adam':
            return AdamW(self.model.parameters(), lr=self.learning_rate, eps=eps)
        else:
            return Adafactor(self.model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=self.learning_rate)
#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from torch.utils.data import Dataset, DataLoader
from transformers import T5TokenizerFast
import pytorch_lightning as pl

class KGQGDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size=16):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.tokenizer = T5TokenizerFast.from_pretrained('t5-base',  extra_ids=0, additional_special_tokens = ['<A>', '<H>', '<R>', '<T>'])
        
    def setup(self, stage=None):   
        self.train_set = KGQGDataset(self.tokenizer, self.data_dir, 'train')
        self.val_set = KGQGDataset(self.tokenizer, self.data_dir, 'val')

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=0)


class KGQGDataset(Dataset):
    def __init__(self, tokenizer, data_dir, split, with_answers=True , max_len=512):
        a = ''
        if with_answers:
            a = 'a_'
        
        self.src_data = data_dir + '/src_' + a + split + '.txt'
        self.tgt_data = data_dir + '/tgt_' + split + '.txt'

        self.max_len = max_len
        self.tokenizer = tokenizer

        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs.input_ids[index]
        target_ids = self.targets.input_ids[index]

        src_mask    = self.inputs.attention_mask[index]
        target_mask = self.targets.attention_mask[index]

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    def _build(self):
        with open(self.src_data, 'r', encoding='utf-8') as f:
            src_text = f.read().splitlines()
        
        with open(self.tgt_data, 'r', encoding='utf-8') as f:
            tgt_text = f.read().splitlines()

        assert len(src_text) == len(tgt_text), "Source and target files are not of same size"
        
        self.inputs = self.tokenizer(src_text, max_length=512, 
                   padding='max_length', truncation=True, return_tensors="pt")
        
        self.targets = self.tokenizer(tgt_text, max_length=512, 
                   padding='max_length', truncation=True, return_tensors="pt")
#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from torch.utils.data import Dataset, DataLoader
from transformers import T5TokenizerFast, BartTokenizerFast
import pytorch_lightning as pl

class KGQGDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size=8, pre_trained='', with_answers=False):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.with_answers = with_answers

        if pre_trained == 't5':
            self.tokenizer = T5TokenizerFast.from_pretrained('t5-base',  extra_ids=0, 
            additional_special_tokens = ['<A>', '<H>', '<R>', '<T>'])
        elif pre_trained == 'bart':
            self.tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base',  extra_ids=0, 
            additional_special_tokens = ['<A>', '<H>', '<R>', '<T>'])
        else:
            raise Exception(f'Unknown pre-trained model {pre_trained}, choose t5 or bart.')

    def setup(self, stage=None):   
        self.train_set = KGQGDataset(self.tokenizer, self.data_dir, 'train', with_answers=self.with_answers)
        self.val_set = KGQGDataset(self.tokenizer, self.data_dir, 'val', with_answers=self.with_answers)
        self.test_set = KGQGDataset(self.tokenizer, self.data_dir, 'test', with_answers=self.with_answers)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=0)


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
        return len(self.inputs.input_ids)

    def __getitem__(self, index):
        source_ids = self.inputs.input_ids[index]
        target_ids = self.targets.input_ids[index]

        src_mask    = self.inputs.attention_mask[index]
        target_mask = self.targets.attention_mask[index]

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    def _build(self):
        with open(self.src_data, 'r', encoding='utf-8') as f:
            src_text = f.read().splitlines()[:2]
        
        with open(self.tgt_data, 'r', encoding='utf-8') as f:
            tgt_text = f.read().splitlines()[:2]

        assert len(src_text) == len(tgt_text), "Source and target files are not of same size"
        
        self.source = src_text
        
        self.inputs = self.tokenizer(src_text, max_length=512, 
                   padding='max_length', truncation=True, return_tensors="pt")
        
        self.targets = self.tokenizer(tgt_text, max_length=512, 
                   padding='max_length', truncation=True, return_tensors="pt")
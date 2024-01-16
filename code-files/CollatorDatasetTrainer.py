import torch
from transformers import Trainer
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, balanced_accuracy_score, roc_auc_score
import numpy as np
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import Trainer
import multiprocessing


class CustomDataCollator:
    def __init__(self, tokenizer, padding_value):
        self.tokenizer = tokenizer
        self.padding_value = padding_value

    def __call__(self, examples):
        batch = {}
        input_ids = pad_sequence([torch.tensor(ids) for ids in examples["input_ids"]], batch_first=True, padding_value=self.padding_value)
        print(input_ids)
        attention_mask = pad_sequence([torch.tensor(mask) for mask in examples["attention_mask"]], batch_first=True, padding_value=self.padding_value)
        token_type_ids = pad_sequence([torch.tensor(ids) for ids in examples["token_type_ids"]], batch_first=True, padding_value=self.padding_value)
        labels = torch.tensor(examples["labels"])
        
        batch["input_ids"] = input_ids
        batch["attention_mask"] = attention_mask
        batch["token_type_ids"] = token_type_ids
        batch["labels"] = labels

        return batch


class Dataset(torch.utils.data.Dataset):    
    def __init__(self, encodings, labels=None):          
        self.encodings = encodings        
        self.labels = labels
     
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.encodings["input_ids"])
   
   
class ToxicTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        weights = torch.tensor([0.5411, 6.5893])
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')
        # compute custom loss
        loss_fct = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


    # def get_train_dataloader(self) -> DataLoader:
    #     if self.train_dataset is None:
    #         raise ValueError("Trainer: training requires a train_dataset.")

    #     data_collator = self.data_collator
    #     train_dataset = self.train_dataset  # self._remove_unused_columns(self.train_dataset, description="training")
    #     train_sampler = self._get_train_sampler()
        
    #     print("create data loader")
    #     data_loader = DataLoader(
    #         train_dataset,
    #         batch_size=self._train_batch_size,
    #         sampler=train_sampler,
    #         collate_fn=data_collator,
    #         drop_last=self.args.dataloader_drop_last,
    #         # num_workers=self.args.dataloader_num_workers,
    #         pin_memory=self.args.dataloader_pin_memory,
    #         # pin_memory=False,
    #         # worker_init_fn=self.args.seed,
    #         num_workers=8, # number of workers is 4 times the number of GPUs
    #     )
        
    #     return  data_loader
    

    
    
    
   

from toxic_tokenizer import Toxic_Tokenizer
import torch
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from transformers import pipeline
from transformers import RobertaConfig, RobertaForMaskedLM, DataCollatorForLanguageModeling
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer , DataCollatorWithPadding , AutoTokenizer, AutoConfig , AutoModelForCausalLM
from transformers import AutoModelForMaskedLM, pipeline, RobertaModel, RobertaTokenizer
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from transformers import (get_linear_schedule_with_warmup , get_cosine_schedule_with_warmup, 
                          get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup)
import datasets
from datasets import Dataset
import os
import argparse
import torch
from bertviz_repo.bertviz import head_view
from CollatorDatasetTrainer import CustomDataCollator
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.utils import class_weight
from CollatorDatasetTrainer import CustomDataCollator , ToxicTrainer
import numpy as np
from utils import *
from utils import split_stratified_into_train_val_test
from sklearn.metrics import balanced_accuracy_score , accuracy_score , precision_score , recall_score , f1_score , roc_auc_score
from bert_loves_chemistry.chemberta.utils.molnet_dataloader import load_molnet_dataset, write_molnet_dataset_for_chemprop
import os
import logging
# remove warnings 
import warnings
import random   
warnings.filterwarnings('ignore')
# Set up logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)




def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_init():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("device inside model init : {}".format(device))
    # config for small model 
    # conf ={"vocab_size": 52_000, "hidden_size": 256, "num_hidden_layers": 2, "num_attention_heads": 2,"max_position_embeddings": 512, "type_vocab_size": 1} 
    # config for medium model
    # conf ={"vocab_size": 52_000, "hidden_size": 512, "num_hidden_layers": 6, "num_attention_heads": 8,"max_position_embeddings": 512, "type_vocab_size": 1}
    # config for large model
    # conf ={"vocab_size": 52_000, "hidden_size": 1024, "num_hidden_layers": 12, "num_attention_heads": 16,"max_position_embeddings": 512, "type_vocab_size": 1}


    config = RobertaConfig(
        vocab_size=512_000,
        max_position_embeddings=512, # max position embedding should be greater than max length of input
        num_attention_heads=4,  #large  # 2 small
        num_hidden_layers=2,  #large # 2 small
        hidden_size=128, # large # 256 small # should be divisible by attention heads
        type_vocab_size=1,
    )

    model = RobertaForMaskedLM(config=config)
    # log number of parameters of the model
    logger.info("number of model parameters: {}".format(model.num_parameters()))
    return model.to(device)


def get_tokenizer(args, vocab_size):
    if args.tokenizer_type == 'Atom-wise':
        # load atom wise tokenizer
        vocab_file=f'vocabs/atom_vocab/{args.tokenizer_data}_{vocab_size}.txt'
        atomwise_tokenizer = Toxic_Tokenizer(vocab_file=vocab_file, tokenizer_type='Atom-wise')
        return atomwise_tokenizer , vocab_size
    elif args.tokenizer_type == 'MacFrag':
        # load chemical rules based tokenizer
        vocab_file = f'vocabs/macFrag_vocab/{args.tokenizer_data}_vocab.smi'
        MacFrag_tokenizer = Toxic_Tokenizer(vocab_file=vocab_file, tokenizer_type='MacFrag')
        vocab_size =  MacFrag_tokenizer.vocab_size
        return MacFrag_tokenizer , vocab_size
    elif args.tokenizer_type == 'Morfessor':
        morf_model_pt= f'models/tokenizers/morfessors/morf_{args.tokenizer_data}_{vocab_size}.bin'
        morf_vocab_pt = f'./vocabs/morf_vocab/morf_{args.tokenizer_data}_{vocab_size}.txt'
        morfessor_tokenizer = Toxic_Tokenizer(vocab_file=morf_vocab_pt,
                                                tokenizer_path=morf_model_pt,
                                                tokenizer_type='Morfessor')
        return morfessor_tokenizer , vocab_size
    elif args.tokenizer_type == 'WordPiece':
        # load Data Driven base tokenizers
        wordpiece_model_pt = f"./models/tokenizers/wordpiece/wordpiece_{args.tokenizer_data}_{vocab_size}.bin"
        wordpiece_vocab_pt = f'vocabs/wordpiece_vocab/{args.tokenizer_data}_{vocab_size}.txt'
        wordpiece_tokenizer = Toxic_Tokenizer(vocab_file= wordpiece_vocab_pt,
                                                tokenizer_path=wordpiece_model_pt,
                                               tokenizer_type='WordPiece')
        return wordpiece_tokenizer , vocab_size
    elif args.tokenizer_type == 'BPE':
        # load Data Driven base tokenizers
        Bpe_model_pt = f'models/tokenizers/bpe/{args.tokenizer_data}_{vocab_size}.bin'
        Bpe_vocab_pt = f'./vocabs/bpe_vocab/{args.tokenizer_data}_{vocab_size}.txt'
        Bpe_tokenizer = Toxic_Tokenizer(vocab_file=Bpe_vocab_pt,
                                                tokenizer_path=Bpe_model_pt,
                                                tokenizer_type='BPE')
        return Bpe_tokenizer , vocab_size
    elif args.tokenizer_type == 'SPE':
        # load Data Driven base tokenizers
        spe_file = f'vocabs/spe_vocab/{args.tokenizer_data}_{vocab_size}'
        vocab_file = f'vocabs/spe_vocab/{args.tokenizer_data}_{vocab_size}.txt'
        spe_tokenizer = Toxic_Tokenizer(vocab_file= vocab_file, spe_file= spe_file, tokenizer_type='SPE')
        return spe_tokenizer , vocab_size
    else:
        raise ValueError('Tokenizer not found')

    
def pre_train(args, vocab_size):
    seed = 42
    set_seed()
    os.environ['TRANSFORMERS_CACHE'] = './cache'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # SET DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: {}".format(device))
    # load tokenizer
    tokenizer , vocab_size = get_tokenizer(args, vocab_size)
    vocab_size = vocab_size
    
    print('device:', device)
    print('Tokenizer type:', tokenizer.tokenizer_type)
    print('Tokenizer type:', args.tokenizer_type)
    print('Tokenizer Data:', args.tokenizer_data)
    print('Tokenizer full Vocab size:', tokenizer.vocab_size)
    print('vocab size of training :', vocab_size)
    print("LM Pre-Train Data :", args.train_data)


    # Prepare datasets for tokenizers
    training_data  = f'./datasets/pre_processed/{args.train_data}'
    # check if path exists
    if os.path.exists(training_data):
       with open(training_data, "r") as f:
            training_data = f.read().strip().split("\n")
       training_data_df = pd.DataFrame(training_data, columns=['smiles'])
    else:
        raise ValueError('Training data not found')
    
    

    # split imbalanced dataset clintox into train, valid, test
    train_df, valid_df = train_test_split(training_data_df, test_size=0.1, random_state=42 ,shuffle=True)
    logging.info(f"train_df shape: {train_df.shape} , valid_df shape: {valid_df.shape}")
    tr_data_dict = {"text": train_df['smiles'].tolist()} 
    val_data_dict = {"text": valid_df['smiles'].tolist()}
    # Create a dataset using the `Dataset` class from `datasets`
    tr_dataset= Dataset.from_dict(tr_data_dict)
    val_dataset= Dataset.from_dict(val_data_dict) 
    # set format to pytorch
    tr_dataset.set_format(type='torch', columns=['text'])
    val_dataset.set_format(type='torch', columns=['text'])
    

    

    logging.info(f'preprocessing {args.train_data} dataset for {tokenizer.tokenizer_type} tokenizer')
    tr_tokenized_dataset = tr_dataset.map(lambda examples: tokenizer(examples["text"], truncation=True,padding=True, max_length=128), batched=True, num_proc=16) # max len 128
    val_tokenized_dataset = val_dataset.map(lambda examples: tokenizer(examples["text"], truncation=True,padding=True, max_length=128), batched=True, num_proc=16) # max len 128
    tr_tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'text'], device=device)
    val_tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'text'], device=device)
    # call add_decoded_tokens to add decoded tokens to the data
    tr_tokenized_dataset = tr_tokenized_dataset.map(lambda batch: add_decoded_tokens(batch= batch, tokenizer=tokenizer))
    val_tokenized_dataset = val_tokenized_dataset.map(lambda batch: add_decoded_tokens(batch= batch, tokenizer=tokenizer))
    logging.info(f"length of train dataset: {len(tr_tokenized_dataset)}")
    logging.info(f"length of valid dataset: {len(val_tokenized_dataset)}")
    

    logging.info(f'Example of a tokenized SMILES from {args.train_data} dataset')
    logging.info(f"Original SMILES: {tr_tokenized_dataset[0]}")

   
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    
    
    model_name = f'{args.train_data}_{args.tokenizer_type}_vocab{vocab_size}_chemBERTa'
    training_args = TrainingArguments(
        output_dir=f'./models/pre_trained_models/{model_name}',          # output directory
        logging_dir=f'./logs/pre_trained_models/{model_name}',          # logging directory
        overwrite_output_dir=True, # overwrite the content of the output directory
        num_train_epochs=30,        # total number of training epochs
        per_device_train_batch_size=16, #256   # batch size per device during training
        per_device_eval_batch_size=16,  #256  # batch size for evaluation
        save_steps=1000,          # number of updates steps before checkpoint saves
        # save_total_limit=2,       # limit the total amount of checkpoints
        logging_steps=100,           # number of updates steps before logging
        evaluation_strategy = "epoch",     # Evaluation strategy to adopt during training
        save_strategy = "epoch", # "epoch" or "steps"",     
        learning_rate=1e-3,                   # Starting learning rate. Can be fine-tuned.
        weight_decay=0.01,                    # Regularization. Can be adjusted.
        seed=seed,             # Seed for the random number generator.
        # load_best_model_at_end=True, # requires save strategy to be "none
        metric_for_best_model="loss",         # Use validation loss to determine the best model.
        # gradient_accumulation_steps=1,        # Number of steps to accumulate gradients before updating weights. Useful if you want to effectively increase batch size without using more memory.
        # dataloader_drop_last=True,
        # dataloader_pin_memory = True,
        #report ton noen    
        report_to = "none",
    )

    model = model_init()
    optimizer = torch.optim.Adam(model.parameters(), lr=training_args.learning_rate)
    warmup_steps = 0.1 * len(tr_tokenized_dataset) * training_args.num_train_epochs
    num_training_steps = len(tr_tokenized_dataset) * training_args.num_train_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)
    trainer = Trainer(
        # model_init=model_init,
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tr_tokenized_dataset,
        eval_dataset=val_tokenized_dataset,
        optimizers=(optimizer, scheduler),
    )
    #Use the PyTorch implementation torch.optim.AdamW instead
    trainer.optimizer = torch.optim.AdamW(trainer.model.parameters())
    trainer.train()
    logging.info(f"Saving model to ./models/pre_trained_models/best_{model_name}")
    trainer.save_model(f'./models/pre_trained_models/best_{model_name}')
    
    eval_results = trainer.evaluate()
    batch_size = training_args.per_device_eval_batch_size
    lr = training_args.learning_rate
    table_path = f'./logs/pre_trained_models/{model_name}_batch_size_{batch_size}_lr_{lr}.csv'
    logging.info(f"Saving eval results to {table_path}")
    perplexity = math.exp(eval_results['eval_loss'])
    logger.info(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    # save eval results and perplexity to csv
    df = pd.DataFrame(eval_results, index=[0])
    df['perplexity'] = perplexity
    df.to_csv(table_path, index=False)




if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description='Pre-train chemBERTa')
    parser.add_argument('--vocab_sizes', nargs='+', type=int, default=None, help='vocab sizes')
    parser.add_argument('--train_data', type=str, default='smilesDB', help='train dataset')
    parser.add_argument('--tokenizer_data', type=str, default='smilesDB', help='tokenizer dataset')
    parser.add_argument('--tokenizer_type', type=str, default='Atom-wise', help='tokenizer type')
    args =  parser.parse_args()
    
    
    if args.vocab_sizes is not None:
        for vocab_size in args.vocab_sizes:
            pre_train(args,vocab_size)
    else:
          pre_train(args , None)

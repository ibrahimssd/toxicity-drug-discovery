from toxic_tokenizer import Toxic_Tokenizer
import torch
import pandas as pd
from transformers import (AutoModelForSequenceClassification, TrainingArguments, Trainer , 
                            DataCollatorWithPadding , EarlyStoppingCallback)
from custom_trainer import ToxicTrainer
from datasets import Dataset
import os
import torch.nn as nn
import argparse
import torch
from sklearn.utils import class_weight
import numpy as np
from utils import *
import os
from transformers import (get_linear_schedule_with_warmup , get_cosine_schedule_with_warmup, 
                          get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup)
import warnings
warnings.filterwarnings('ignore')
import logging
import random
import selfies as sf
from rdkit import Chem
from rdkit.Chem import EnumerateStereoisomers
from sklearn.metrics import roc_auc_score
# Set up logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)



DATASET_PATHS = {
    'clintox': './datasets/scaffold_splits/clintox_{}.csv',
    'tox21': './datasets/scaffold_splits/tox21_{}.csv',
    'hips': './datasets/HIPS/hips_{}.csv'
}


def bootstrap_confidence_interval(data, stat_function=np.mean, alpha=0.05, n_bootstrap=10000):
    """
    Calculate the confidence interval of a given metric using bootstrapping.
    
    :param data: 1-D array-like data points.
    :param stat_function: Function to apply to the resampled data.
    :param alpha: Significance level.
    :param n_bootstrap: Number of bootstrap samples to create.
    :return: Tuple containing the lower and upper bound of the confidence interval.
    """
    bootstrapped_stats = []
    n = len(data)
    for _ in range(n_bootstrap):
        resample = np.random.choice(data, n, replace=True)
        stat = stat_function(resample)
        bootstrapped_stats.append(stat)
    
    lower = np.percentile(bootstrapped_stats, 100 * (alpha / 2))
    upper = np.percentile(bootstrapped_stats, 100 * (1 - alpha / 2))
    return lower, upper



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def load_dataframes(target_data):
    if target_data not in DATASET_PATHS:
        raise ValueError('Train dataset not found')
    train_df = pd.read_csv(DATASET_PATHS[target_data].format('train'))
    valid_df = pd.read_csv(DATASET_PATHS[target_data].format('val'))
    test_df = pd.read_csv(DATASET_PATHS[target_data].format('test'))
    logger.info(f"columns names of {target_data} target dataset: {train_df.columns}")
    return train_df, valid_df, test_df

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


def model_init(check_pt):
    num_labels=2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(check_pt, num_labels=num_labels, use_auth_token=False)
    return model.to(device)


def train_auxiliary_model(args, vocab_size):
    logger.info("Training auxiliary model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: {}".format(device))
    logger.info(" Aux task: {}".format(args.aux_task))
    logger.info(" tokenizer type: {}".format(args.tokenizer_type))
    logger.info(" vocab size: {}".format(vocab_size))
    logger.info(" tokenizer data: {}".format(args.tokenizer_data))
    logger.info(" pretrained data: {}".format(args.pretrained_data))
    logger.info(" target dataset to train LM: {}".format(args.target_data))
    logger.info(" training mol representation: {}".format(args.mol_rep))
    
    seed = 40
    set_seed(seed)
    # load tokenizer
    tokenizer , tok_vocab_size = get_tokenizer(args, vocab_size)
    # Load Evaluating Data
    train_df, valid_df, test_df = load_dataframes(args.target_data)
    vocab_size = vocab_size
    # LENGHT before droping duplicates
    logger.info("train_df shape: " + str(train_df.shape))
    logger.info("valid_df shape: " + str(valid_df.shape))
    logger.info("test_df shape: " + str(test_df.shape))
    # drop duplicates
    train_df = train_df.drop_duplicates(subset=[args.mol_rep])
    valid_df = valid_df.drop_duplicates(subset=[args.mol_rep])
    test_df = test_df.drop_duplicates(subset=[args.mol_rep])
    
    
    tr_text= train_df[args.mol_rep].tolist()
    val_text= valid_df[args.mol_rep].tolist()
    tr_labels= train_df[args.aux_task].values.tolist()
    val_labels= valid_df[args.aux_task].values.tolist()
    # Add test data to validation data
    val_text.extend(test_df[args.mol_rep].tolist())
    val_labels.extend(test_df[args.aux_task].values.tolist())
    logging.info("labels counts before Augmenting " + str(np.unique(tr_labels, return_counts=True)))

    # REMOVE AUGMENTATION FOR NOW , LEAVE AUGMENTATION FOR PRE-TRAINING
    if args.aug:
        logger.info("Augmenting training data")
        if args.mol_rep == 'smiles':
            tr_text, tr_labels = augment_smiles(tr_text, tr_labels, augmentation_factor=10) 
        elif args.mol_rep == 'selfies':
            tr_text, tr_labels = augment_selfies(tr_text, tr_labels, augmentation_factor=10)
    
    
    
    # log the new weights after augmentation
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(tr_labels, return_counts=True)[0], y=tr_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float)  
    logger.info(" New class weights " + str(class_weights))
    logging.info("labels counts after Augmenting " + str(np.unique(tr_labels, return_counts=True)))

    # Create a dictionary with the text data
    tr_data_dict = {"text": tr_text,  
                "labels":  tr_labels}        
    val_data_dict = {"text": val_text ,  
                    "labels":  val_labels}   
    # Create a dataset using the `Dataset` class from `datasets`
    tr_dataset= Dataset.from_dict(tr_data_dict)
    val_dataset= Dataset.from_dict(val_data_dict) 

    tr_tokenized_dataset = tr_dataset.map(lambda examples: tokenizer(examples["text"], truncation=True,padding=True, max_length=128), batched=True, num_proc=16)
    val_tokenized_dataset = val_dataset.map(lambda examples: tokenizer(examples["text"], truncation=True,padding=True, max_length=128), batched=True, num_proc=16)
    tr_tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'text'])
    val_tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'text'])
    # call add_decoded_tokens to add decoded tokens to the data
    tr_tokenized_dataset = tr_tokenized_dataset.map(lambda batch: add_decoded_tokens(batch= batch, tokenizer=tokenizer))
    val_tokenized_dataset = val_tokenized_dataset.map(lambda batch: add_decoded_tokens(batch= batch, tokenizer=tokenizer))

    logger.info("Tokenized dataset example in Aux Task:" + str(tr_tokenized_dataset[2]))
    
    data_collator= DataCollatorWithPadding(tokenizer=tokenizer, padding ="longest", max_length=128)
    pre_trained_model = f'{args.pretrained_data}_{args.tokenizer_type}_vocab{vocab_size}_chemBERTa'
    logger.info("Pre-Trained model: " + str(pre_trained_model))
    check_point =  f'./models/pre_trained_models/best_{pre_trained_model}'
    # make a global checkpoint to load the model
    aux_classifier = model_init(check_pt=check_point)
    logger.info("Pre-Trained model vocab size:" + str(aux_classifier.config.vocab_size))
    aux_classifier_name = f'{args.tokenizer_type}_{args.aux_task}_{args.mol_rep}_{args.target_data}_{vocab_size}'
    training_args = TrainingArguments(
        output_dir="./models/aux_classifiers/"+aux_classifier_name,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        save_steps=2000,
        logging_steps=100,
        evaluation_strategy = "epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        save_strategy = "epoch",
        seed=seed,
        learning_rate=1e-4,                   # Starting learning rate. Can be fine-tuned.
        weight_decay=0.01,                    # Strength of weight decay
        gradient_accumulation_steps=1,        # Number of updates steps to accumulate before performing a backward/update pass.
        report_to="none",
        metric_for_best_model = 'eval_AUC_ROC_delong',
        )
    
    optimizer = torch.optim.Adam(aux_classifier.parameters(), lr=training_args.learning_rate)
    aux_classifier.optimizer = optimizer
    # cosine scheduler
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=1000)
    # linear scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=1000)
    # Define the Trainer
    trainer = ToxicTrainer(
        model=aux_classifier,
        args=training_args,
        train_dataset=tr_tokenized_dataset,
        eval_dataset= val_tokenized_dataset,
        data_collator=data_collator,
        optimizers=(optimizer, scheduler),
        # compute_metrics= lambda predictions: weighted_compute_metrics(predictions, class_weights),
        compute_metrics= compute_metrics,
        class_weights=class_weights
    )
   
    # log number of trainable parameters
    total_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    logger.info("Number of trainable parameters : {}".format(total_params))
    # Train the model
    trainer.train()
    # save model
    aux_classifier_checkpoint = f'./models/aux_classifiers/best_{aux_classifier_name}'
    trainer.save_model(aux_classifier_checkpoint)

    # log best f1 score
    logger.info("Best AUC_ROC Delong score: {}".format(trainer.state.best_metric))

    # return the trained model
    return aux_classifier , aux_classifier_checkpoint   



def fine_tune(args, vocab_size, aux_classifier, aux_classifier_checkpoint):
    seed = 40
    set_seed(seed)
    os.environ['TRANSFORMERS_CACHE'] = './cache'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: {}".format(device))
    # load tokenizer
    tokenizer , vocab_size = get_tokenizer(args, vocab_size)
   
    logger.info("Tokenizer type: {}".format(tokenizer.tokenizer_type))
    logger.info("Tokenizer Full Vocab size(Special Tokens Included): {}".format(tokenizer.vocab_size))
    logger.info("Training Vocab Size : {}".format(vocab_size))
    logger.info("tokenizer data: {}".format(args.tokenizer_data))  
    logger.info("pretrained data: {}".format(args.pretrained_data))
    logger.info("target dataset to train LM: {}".format(args.target_data))
    logger.info(" training mol representation: {}".format(args.mol_rep))
    logger.info("Processing task: {}".format(args.task))
    
   
    # Load Evaluating Data
    train_df, valid_df, test_df = load_dataframes(args.target_data)
    train_df = train_df.drop_duplicates(subset=[args.mol_rep])
    valid_df = valid_df.drop_duplicates(subset=[args.mol_rep])
    test_df = test_df.drop_duplicates(subset=[args.mol_rep])
    tr_text= train_df[args.mol_rep].tolist()
    tr_labels= train_df[args.task].values.tolist()
    val_text= valid_df[args.mol_rep].tolist()
    val_labels= valid_df[args.task].values.tolist()
    te_text= test_df[args.mol_rep].tolist()
    te_labels= test_df[args.task].values.tolist()
    
    # add test data to validation data
    val_text.extend(te_text)
    val_labels.extend(te_labels)

    # class weights for imbalanced dataset
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(tr_labels, return_counts=True)[0], y=tr_labels)
    logging.info("labels counts " + str(np.unique(train_df[args.task], return_counts=True)))
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    
    logging.info("class weights " + str(class_weights))
    logger.info("train_df shape: " + str(train_df.shape))
    logger.info("valid_df shape: " + str(valid_df.shape))
    logger.info("test_df shape: " + str(test_df.shape))
    

    # log the new weights after augmentation
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(tr_labels, return_counts=True)[0], y=tr_labels)
    logging.info("labels counts " + str(np.unique(tr_labels, return_counts=True)))
    class_weights = torch.tensor(class_weights, dtype=torch.float)  
    logger.info(" New class weights " + str(class_weights))
    # Create a dictionary with the text data
    tr_data_dict = {"text": tr_text,  
                "labels":  tr_labels}        
    val_data_dict = {"text": val_text ,  
                    "labels":  val_labels}   
    # Create a dataset using the `Dataset` class from `datasets`
    tr_dataset= Dataset.from_dict(tr_data_dict)
    val_dataset= Dataset.from_dict(val_data_dict) 

    tr_tokenized_dataset = tr_dataset.map(lambda examples: tokenizer(examples["text"], truncation=True,padding=True, max_length=256), batched=True, num_proc=16)
    val_tokenized_dataset = val_dataset.map(lambda examples: tokenizer(examples["text"], truncation=True,padding=True, max_length=256), batched=True, num_proc=16)
    tr_tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'text'])
    val_tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'text'])
    # call add_decoded_tokens to add decoded tokens to the data
    tr_tokenized_dataset = tr_tokenized_dataset.map(lambda batch: add_decoded_tokens(batch= batch, tokenizer=tokenizer))
    val_tokenized_dataset = val_tokenized_dataset.map(lambda batch: add_decoded_tokens(batch= batch, tokenizer=tokenizer))

    logger.info("train tokenized dataset example:" + str(tr_tokenized_dataset[2]))
    data_collator= DataCollatorWithPadding(tokenizer=tokenizer, padding ="longest", max_length=256)
    pre_trained_model = f'{args.pretrained_data}_{args.tokenizer_type}_vocab{vocab_size}_chemBERTa'
    logger.info("Pre-Trained model: " + str(pre_trained_model))
    check_point =  f'./models/pre_trained_models/best_{pre_trained_model}'
    # make a global checkpoint to load the model
    
    classifier =  model_init(check_pt=aux_classifier_checkpoint)
    

    logger.info("Pre-Trained model vocab size:" + str(classifier.config.vocab_size))
    classifier_name = f'{args.tokenizer_type}_{args.task}_{args.mol_rep}_{args.target_data}_{vocab_size}'

    training_args = TrainingArguments(
        output_dir="./models/classifiers/"+classifier_name,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        save_steps=2000,
        logging_steps=100,
        evaluation_strategy = "epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        save_strategy = "epoch",
        seed=seed,
        learning_rate=1e-4,                   # Starting learning rate. Can be fine-tuned.
        weight_decay=0.01,                    # Strength of weight decay
        gradient_accumulation_steps=1,        # Number of updates steps to accumulate before performing a backward/update pass.
        report_to="none",
        metric_for_best_model = 'eval_AUC_ROC_delong',
        greater_is_better = True
        )

    early_stopping = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)
    num_training_steps = len(tr_tokenized_dataset) // training_args.per_device_train_batch_size * training_args.num_train_epochs
    warmup_steps = int(num_training_steps * 0.1)  # 10% of train steps as warm-up
    optimizer = torch.optim.Adam(classifier.parameters(), lr=training_args.learning_rate)
    classifier.optimizer = optimizer    
    # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)
    classifier.dropout = nn.Dropout(0.1)       
    # Define the Trainer
    trainer = ToxicTrainer(
        # model_init=model_init,
        model=classifier,
        args=training_args,
        train_dataset=tr_tokenized_dataset,
        eval_dataset= val_tokenized_dataset,
        data_collator=data_collator,
        # compute_metrics= lambda predictions: weighted_compute_metrics(predictions, class_weights),
        compute_metrics= compute_metrics,
        # callbacks=[early_stopping],
        optimizers=(optimizer, scheduler),
        class_weights=class_weights
    )
    
    # log number of trainable parameters before freezing
    total_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    logger.info("Number of trainable parameters before freezing: {}".format(total_params))
    
    # freeze the pre-trained model weights
    # for param in trainer.model.base_model.parameters():
    #     param.requires_grad = False

    # log number of trainable parameters
    total_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    logger.info("Number of trainable parameters after freezing: {}".format(total_params)) 
    # Train the model
    trainer.train()
    # save model
    trainer.save_model(f'./models/classifiers/best_{classifier_name}')

    # LOAD BEST MODEL
    # trainer.model = model_init(check_pt="./models/classifiers/"+classifier_name)
    plot_path = "./plots/classification/"
    
    plot_training_progress_metrics(trainer , classifier_name, plot_path= plot_path)
    metrics_values = plot_combined_evaluation_metrics(trainer , classifier_name,plot_path= plot_path)
    plot_final_evaluation_metrics_points(trainer , classifier_name, plot_path= plot_path)
    # plot_final_evaluation_metrics_bars(trainer , classifier_name, plot_path= plot_path)
    
    # EXTRACT  SCORES
    validation_results = trainer.evaluate()
    logger.info("Validation results: {}".format(validation_results))
    auc_roc = validation_results['eval_AUC_ROC']
    auc_roc_delong = validation_results['eval_AUC_ROC_delong']
    auc_roc_ci_delong = validation_results['eval_AUC_ROC_ci_delong']
    logger.info("AUC ROC: {}".format(auc_roc))
    logger.info("AUC ROC Delong: {}".format(auc_roc_delong))
    logger.info("Best Score: {}".format(trainer.state.best_metric))
    logger.info("AUC ROC Delong CI: {}".format(auc_roc_ci_delong))
    auc_roc_delong_lb , auc_roc_delong_ub = auc_roc_ci_delong 
    auc_prc = validation_results['eval_AUC_PRC']
    f1= validation_results['eval_f1']
    balanced_accuracy = validation_results['eval_balanced_accuracy_score']
    accuracy= validation_results['eval_accuracy']
    # confidence interval for the metrics 
    auc_roc_samples = metrics_values['eval_AUC_ROC']
    auc_prc_samples = metrics_values['eval_AUC_PRC']
    f1_samples = metrics_values['eval_f1']
    balanced_accuracy_samples = metrics_values['eval_balanced_accuracy_score']
    accuracy_samples = metrics_values['eval_accuracy']
    AUC_ROC_delong_samples = metrics_values['eval_AUC_ROC_delong']
    # CI for the metrics
    auc_roc_lb, auc_roc_ub = bootstrap_confidence_interval(auc_roc_samples)
    auc_prc_lb, auc_prc_ub = bootstrap_confidence_interval(auc_prc_samples)
    f1_lb, f1_ub = bootstrap_confidence_interval(f1_samples)
    balanced_accuracy_lb, balanced_accuracy_ub = bootstrap_confidence_interval(balanced_accuracy_samples)
    accuracy_lb, accuracy_ub = bootstrap_confidence_interval(accuracy_samples)


    batch_size = training_args.per_device_train_batch_size
    lr = training_args.learning_rate
    epochs = training_args.num_train_epochs
    
    
    table_path = f'./logs/fine_tune/auc_scores_batch{batch_size}_lr{lr}_epochs{epochs}_task{args.task}_aux_task{args.aux_task}_AugmentationAux{args.aug}_resultes.csv'
    
    if not os.path.exists(table_path):
        logging_dataframe = pd.DataFrame(columns=['tokenizer_type','mol_rep','task','target_data','vocab_size',
                                    'auc_roc','auc_roc_lb', 'auc_roc_ub',
                                    'auc_roc_delong', 'auc_roc_delong_lb', 'auc_roc_delong_ub',
                                    'auc_prc','auc_prc_lb', 'auc_prc_ub',
                                    'f1','f1_lb', 'f1_ub',
                                    'balanced_accuracy', 'balanced_accuracy_lb', 'balanced_accuracy_ub' ,
                                    'accuracy', 'accuracy_lb', 'accuracy_ub',
                                    'auc_roc_samples', 'auc_prc_samples', 'f1_samples', 'balanced_accuracy_samples', 'accuracy_samples','auc_roc_delong_samples' 
                                        ], index=None)
        logging_dataframe.to_csv(table_path, index=False)
       
    # append the new row
    logging_dataframe = pd.read_csv(table_path)
    new_row = {'tokenizer_type': args.tokenizer_type, 'mol_rep': args.mol_rep, 'task': args.task, 'target_data': args.target_data, 'vocab_size': vocab_size,
               'auc_roc': auc_roc, 'auc_roc_lb': auc_roc_lb, 'auc_roc_ub': auc_roc_ub,
               'auc_roc_delong': auc_roc_delong, 'auc_roc_delong_lb': auc_roc_delong_lb, 'auc_roc_delong_ub': auc_roc_delong_ub,
               'auc_prc': auc_prc, 'auc_prc_lb': auc_prc_lb, 'auc_prc_ub': auc_prc_ub,
               'f1': f1, 'f1_lb': f1_lb, 'f1_ub': f1_ub,
                'balanced_accuracy': balanced_accuracy,  'balanced_accuracy_lb': balanced_accuracy_lb, 'balanced_accuracy_ub': balanced_accuracy_ub,
                'accuracy': accuracy, 'accuracy_lb': accuracy_lb, 'accuracy_ub': accuracy_ub,
                'auc_roc_samples': auc_roc_samples, 'auc_prc_samples': auc_prc_samples, 'f1_samples': f1_samples, 'balanced_accuracy_samples': balanced_accuracy_samples, 'accuracy_samples': accuracy_samples, 'auc_roc_delong_samples': AUC_ROC_delong_samples
                }
    
    logging_dataframe = logging_dataframe.append(new_row, ignore_index=True)
    logging_dataframe.to_csv(table_path, index=False)
   
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pre-train chemBERTa')
    parser.add_argument('--vocab_sizes', nargs='+', type=int, default=None, help='vocab sizes')
    parser.add_argument('--target_data', type=str, default='clintox', help='target dataset')
    parser.add_argument('--pretrained_data', type=str, default='smilesDB', help='pretrained dataset')
    parser.add_argument('--mol_rep' , type=str, default='smiles', help='molecular representation')
    parser.add_argument('--task', type=str, default='CT_TOX', help='target task')
    parser.add_argument('--aux_task', type=str, default='FDA_APPROVED', help='auxiliary task')
    parser.add_argument('--tokenizer_data', type=str, default='smilesDB', help='tokenizer data')
    parser.add_argument('--tokenizer_type', type=str, default='BPE', help='tokenizer type')
    parser.add_argument('--epochs', type=int, default=15, help='number of epochs')
    parser.add_argument('--aug', action='store_true', help='augment training data')
    parser.add_argument('--no-aug', action='store_false', dest='aug', help='do not augment training data')
    args = parser.parse_args()
    
    # print aug flag
    logger.info("Aug flag: {}".format(args.aug))
    if args.vocab_sizes is not None:
        for vocab_size in args.vocab_sizes:
             aux_classifier , aux_classifier_checkpoint = train_auxiliary_model(args, vocab_size)
             fine_tune(args, vocab_size, aux_classifier, aux_classifier_checkpoint)
    else:   
            aux_classifier , aux_classifier_checkpoint = train_auxiliary_model(args, None)
            fine_tune(args, None, aux_classifier, aux_classifier_checkpoint)
    
   
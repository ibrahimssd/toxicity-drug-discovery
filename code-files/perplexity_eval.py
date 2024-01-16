from toxic_tokenizer import Toxic_Tokenizer
import torch
from transformers import  AutoModelForMaskedLM
import os
import argparse
import torch
import math
import numpy as np
import torch.nn as nn
from utils import *
from datasets import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')
import logging
import random
# Set up logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)




DATASET_PATHS = {
    'clintox': './datasets/scaffold_splits/clintox_{}.csv',
    'tox21': './datasets/scaffold_splits/tox21_{}.csv',
    'hips': './datasets/HIPS/hips_{}.csv'
}


def load_dataframes(target_data):
    if target_data not in DATASET_PATHS:
        raise ValueError('Train dataset not found')
    train_df = pd.read_csv(DATASET_PATHS[target_data].format('train'))
    valid_df = pd.read_csv(DATASET_PATHS[target_data].format('val'))
    test_df = pd.read_csv(DATASET_PATHS[target_data].format('test'))
    logger.info(f"columns names of {target_data} target dataset: {train_df.columns}")
    return train_df, valid_df, test_df




def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
        # vocab_size =  MacFrag_tokenizer.vocab_size
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


def prepare_dataset(args, tokenizer, device='cuda'):
    # Prepare datasets for tokenizers
    training_data  = f'./datasets/pre_processed/{args.test_data}'
    # check if path exists
    if os.path.exists(training_data):
       with open(training_data, "r") as f:
            training_data = f.read().strip().split("\n")
       training_data_df = pd.DataFrame(training_data, columns=[args.mol_rep])
       logger.info(f"size of {args.test_data} dataset: {len(training_data_df)}")
    else:
        raise ValueError('Training data not found')
    
    logger.info(f"training df samples: {training_data_df.head()}")

    train_df, valid_df = train_test_split(training_data_df, test_size=0.1, random_state=42 ,shuffle=True)
    
    
    logging.info(f"train_df shape: {train_df.shape} , valid_df shape: {valid_df.shape}")
    tr_data_dict = {"text": train_df[args.mol_rep].tolist()} 
    val_data_dict = {"text": valid_df[args.mol_rep].tolist()}
    # Create a dataset using the `Dataset` class from `datasets`
    tr_dataset= Dataset.from_dict(tr_data_dict)
    val_dataset= Dataset.from_dict(val_data_dict) 
    # set format to pytorch
    tr_dataset.set_format(type='torch', columns=['text'])
    val_dataset.set_format(type='torch', columns=['text'])
    

    logging.info(f'preprocessing {args.test_data} dataset for {tokenizer.tokenizer_type} tokenizer')
    tr_tokenized_dataset = tr_dataset.map(lambda examples: tokenizer(examples["text"], truncation=True,padding=True, max_length=128), batched=True, num_proc=32) # max len 128
    val_tokenized_dataset = val_dataset.map(lambda examples: tokenizer(examples["text"], truncation=True,padding=True, max_length=128), batched=True, num_proc=32) # max len 128
    tr_tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'text'], device=device)
    val_tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'text'], device=device)
    # call add_decoded_tokens to add decoded tokens to the data
    tr_tokenized_dataset = tr_tokenized_dataset.map(lambda batch: add_decoded_tokens(batch= batch, tokenizer=tokenizer))
    val_tokenized_dataset = val_tokenized_dataset.map(lambda batch: add_decoded_tokens(batch= batch, tokenizer=tokenizer))
    logging.info(f"length of train dataset: {len(tr_tokenized_dataset)}")
    logging.info(f"length of valid dataset: {len(val_tokenized_dataset)}")
    
    logging.info(f'Example of a tokenized SMILES from {args.test_data} dataset')
    logging.info(f"Original SMILES: {tr_tokenized_dataset[0]}")
    logging.info(f"Tokenized SMILES: {tr_tokenized_dataset[0]['decoded_tokens']}")
    return tr_tokenized_dataset, val_tokenized_dataset




def loss_perplexity(model, tokenizer, sentence):
    # tokens 
    # logger.info("sentence: {}".format(sentence))
    # logger.info("tokens : {}".format(tokenizer.tokenize(sentence)))
    
    tensor_input = tokenizer.encode(sentence, return_tensors='pt')
    repeat_input = tensor_input.repeat(tensor_input.size(-1)-2, 1)
    mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
    masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
    labels = repeat_input.masked_fill(masked_input != tokenizer.mask_token_id, -100)
    # make sure both labels and masked_input has same size
    labels = labels[:, :masked_input.size(-1)]    
    # logger.info("Masked input: {}".format(tokenizer.decode(masked_input[0])))
    with torch.inference_mode():
        loss = model(masked_input, labels=labels).loss
    
    # number of masked tokens
    num_masked_tokens = (labels != -100).sum().item()

    perplexity = math.exp(loss.item())
    return loss.item() ,  perplexity , num_masked_tokens


def sentence_loss(model, tokenizer, sentence):
    with torch.no_grad():
            model.eval()
            # Load pre-trained model tokenizer (vocabulary)
            tokenize_input = tokenizer.tokenize(sentence)
            tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
            sen_len = len(tokenize_input)
            mask_token = tokenizer.mask_token
            sentence_loss = 0

            for i, word in enumerate(tokenize_input):
                # add mask to i-th character of the sentence
                tokenize_input[i] = mask_token
                mask_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])

                output = model(mask_input)

                prediction_scores = output[0]
                softmax = nn.Softmax(dim=0)
                ps = softmax(prediction_scores[0, i]).log()
                word_loss = ps[tensor_input[0, i]]
                sentence_loss += word_loss.item()

                tokenize_input[i] = word

            ppl = np.exp(-sentence_loss/sen_len)
            
    return sentence_loss, ppl



def eval(args, vocab_size):
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
    logger.info("Language pretrained data: {}".format(args.pretrained_data))
    logger.info("Test data: {}".format(args.test_data))
    logger.info("Target data: {}".format(args.target_data))
    logger.info("Molecular representation: {}".format(args.mol_rep))
    
    
    
    
    # prepare dataset
    # tr_tokenized_dataset, val_tokenized_dataset = prepare_dataset(args, tokenizer, device)
    # logger.info(' val_tokenized_dataset: {}'.format(val_tokenized_dataset[0]))

    # LOAD DATA
    train_df, valid_df, test_df = load_dataframes(args.target_data)
    logger.info(f"train_df shape: {train_df.shape} , valid_df shape: {valid_df.shape} , test_df shape: {test_df.shape}")
    # combine validation and test data
    evaluation_df  = pd.concat([valid_df, test_df], axis=0)

    if args.target_data == 'hips':
        evaluation_df = pd.concat([evaluation_df, train_df], axis=0)
    


    # load best model
    pre_trained_model = f'{args.pretrained_data}_{args.tokenizer_type}_vocab{vocab_size}_chemBERTa'
    logger.info("Pre-Trained Language model: " + str(pre_trained_model))
    check_point =  f'./models/pre_trained_models/best_{pre_trained_model}'
    Language_model = AutoModelForMaskedLM.from_pretrained(check_point)
    


    # calculate  perplexity per word for the validation set using the best model
    # calculate the number of words in the validation set
    # test_sentences =  val_tokenized_dataset['text'] + tr_tokenized_dataset['text']
    evaluation_sentences = evaluation_df[args.mol_rep].tolist()
    logger.info("length of test sentences: {}".format(len(evaluation_sentences)))
    
    

    val_words_with_special_tokens = 0
    val_words_without_special_tokens = 0

    val_nll = []
    perplexities = []
    sen_losses = []
    number_of_words = 0
    sen_perplexities = []
    for sentence in evaluation_sentences:
        tokens_with_special_tokens = tokenizer.decode(tokenizer.encode(sentence))
        tokens_without_special_tokens = tokenizer.tokenize(sentence)
        val_words_with_special_tokens += len(tokens_with_special_tokens)
        val_words_without_special_tokens += len(tokens_without_special_tokens)

        sen_loss , sen_perplexity = sentence_loss(Language_model, tokenizer, sentence)
        sen_losses.append(sen_loss)
        sen_perplexities.append(sen_perplexity)
        # logger.info(" sen loss , perplexity: {}, {}".format(sen_loss, sen_perplexity))


        loss , perplexity , num_masked_tokens = loss_perplexity(Language_model, tokenizer, sentence)
        val_nll.append(loss)
        perplexities.append(perplexity)
        number_of_words += num_masked_tokens
        logger.info(f'loss: {loss} , perplexity: {perplexity} , num_masked_tokens: {num_masked_tokens}')
        
        

    # calculate perplexity per word without special tokens
    logger.info("val_words_without_special_tokens: {}".format(val_words_without_special_tokens))
    logger.info(" number of words: {}".format(number_of_words))
    logger.info("val words with special tokens: {}".format(val_words_with_special_tokens))
    logger.info(f'perplexity per word (perplexity_sen_loss_minus): {math.exp(-sum(sen_losses) / val_words_without_special_tokens)} , vocab size: {vocab_size}')
    logger.info(f'perplexity per word (perplexity_sen_loss_plus): {math.exp(sum(sen_losses) / val_words_without_special_tokens)} , vocab size: {vocab_size}')
    logger.info(f'perplexity per word : {math.exp(sum(val_nll) / val_words_without_special_tokens)} , vocab size: {vocab_size}')
    logger.info(f'perplexity per word : {math.exp(sum(val_nll) / number_of_words)} , vocab size: {vocab_size}')
    logger.info(f'perplexity per word without special tokens: {math.exp(-sum(val_nll) / number_of_words)} , vocab size: {vocab_size}')
    logger.info(f'perplexity per word without special tokens: {sum(perplexities) / number_of_words  } , vocab size: {vocab_size}')
    logger.info(f'perplexity average: {np.mean(perplexities)} , vocab size: {vocab_size}')

    perplexity_per_word = math.exp(sum(val_nll) / number_of_words)
    perplexity_sen_loss_minus = math.exp(-sum(sen_losses) / val_words_without_special_tokens)
    perplexity_sen_loss_plus = math.exp(sum(sen_losses) / val_words_without_special_tokens)
    # perplexities avg 
    perplexity_avg =  np.mean(perplexities)
    
    # save perplexity per word in csv file
    path = f'./logs/perplexity/perplexity_per_word_{args.target_data}.csv'
    if not os.path.exists(path):
        perplexity_per_word_df = pd.DataFrame(columns=['tokenizer_type', 'vocab_size', 'perplexity_per_word', 'perplexity_avg',
                                                'perplexity_sen_loss_minus', 'perplexity_sen_loss_plus',
                                                'perplexities', 'sen_perplexities'], index=None)
        
        perplexity_per_word_df.to_csv(path, index=False)

    
    perplexity_per_word_df = pd.read_csv(path)
    perplexity_per_word_df = perplexity_per_word_df.append({'tokenizer_type': args.tokenizer_type, 'vocab_size': vocab_size, 'perplexity_per_word': perplexity_per_word, 'perplexity_avg': perplexity_avg,
                                                'perplexity_sen_loss_minus': perplexity_sen_loss_minus, 'perplexity_sen_loss_plus': perplexity_sen_loss_plus,
                                                'perplexities': perplexities, 'sen_perplexities': sen_perplexities}, ignore_index=True)
    perplexity_per_word_df.to_csv(path, index=False)

    
        
        # filter out the perplexity per word for the BPE tokenizer and perplexity per word columns 
        # perplexity_per_word_df = perplexity_per_word_df[perplexity_per_word_df['tokenizer_type'] == 'BPE']
        # perplexity_per_word_df = perplexity_per_word_df[['tokenizer_type', 'vocab_size', 'perplexity_per_word']] 
        # perplexity_per_word_df['perplexities'] = perplexity_per_word_df['perplexities'].apply(lambda x: x.strip('][').split(', '))
        # perplexity_per_word_df['perplexities'] = perplexity_per_word_df['perplexities'].apply(lambda x: [float(i) for i in x])
        # perplexity_per_word_df['avg_perplexities'] = perplexity_per_word_df['perplexities'].apply(lambda x: sum(x) / len(x))
    


def plot_perplexity_vs_vocab(csv_path, test_data):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)    
    perplexity_atomic_smiles  = df[(df['tokenizer_type'] == 'Atom-wise') & (df['vocab_size'] == 163)]['perplexity_per_word'].values[0]
    # perplexity_atomic_selfies = df[(df['tokenizer_type'] == 'Atom-wise') & (df['vocab_size'] == 204)]['perplexity_per_word'].values[0]

    
    # Filter the DataFrame for the tokenizer type , perplexity per word and vocab size columns
    df_filtered = df[['tokenizer_type', 'vocab_size', 'perplexity_per_word']]
    logger.info(f"df_filtered: {df_filtered}")
    
    
    # remove atomic tokenizer from the dataframe
    df = df[df['tokenizer_type'] != 'Atom-wise']
    df = df[df['tokenizer_type'] != 'MacFrag']


    # Ensure that 'vocab_size' is an integer for sorting
    df['vocab_size'] = df['vocab_size'].astype(int)

    # Sort the DataFrame by 'vocab_size' to ensure the plot lines are ordered
    df.sort_values(by='vocab_size', inplace=True)

    # Plot perplexity for each tokenizer type
    plt.figure(figsize=(10, 6))
    tokenizer_types = df['tokenizer_type'].unique()
    
    for tokenizer in tokenizer_types:
        # Filter the DataFrame for each tokenizer type
        tokenizer_df = df[df['tokenizer_type'] == tokenizer]
        
        # Plot each tokenizer type with a different line
        plt.plot(tokenizer_df['vocab_size'], tokenizer_df['perplexity_per_word'], label=tokenizer, marker='o')
    
    # Plot horizontal lines for atomic SMILES and SELFIES
    plt.axhline(y=perplexity_atomic_smiles, color='gray', linestyle='--', label='Atomic SMILES')
    # plt.axhline(y=perplexity_atomic_selfies, color='g', linestyle='--', label='Atomic SELFIES')

    # Log scale for x-axis to see the changes more clearly
    plt.xscale('log')
    plt.xlabel('Vocabulary Size', fontsize=14, fontweight='bold')
    plt.ylabel('Perplexity per Fragment', fontsize=14, fontweight='bold')
    plt.title(f'Vocabulary Size Impact on Per-Fragment Perplexity Across Various Tokenizers Evaluated on {test_data} Dataset')


    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot as a PNG file
    logger.info(f"save plot at: ./plots/perplexity/perplexity_vs_vocab_size_{test_data}.png")
    plt.savefig(f'./plots/perplexity/perplexity_vs_vocab_size_{test_data}.png')

def plot_auc_comparison(csv_path):
    # Load the data from CSV file
    df = pd.read_csv(csv_path)
    df = df[['tokenizer_type','mol_rep', 'vocab_size', 'task', 'auc_roc_delong', 'auc_roc_delong_lb', 'auc_roc_delong_ub']]

    logger.info(f"df: {df.head()}")

    # Filter the data for SELFIES and SMILES representations
    df_smiles = df[df['mol_rep'] == 'smiles']
    df_selfies = df[df['mol_rep'] == 'selfies']
    
    
    # Find unique tasks to plot
    tasks = df['task'].unique()

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))

    # Width of the bars
    bar_width = 0.35
    # Set positions of the bars
    index = np.arange(len(tasks))

    # Define error bar format
    error_kw = {'capsize': 5, 'capthick': 2, 'ecolor': 'black'}

    # Loop over tasks to plot AUC-ROC for SMILES and SELFIES
    for i, task in enumerate(tasks):
        # Get AUC-ROC and confidence intervals for SMILES
        smiles_auc = df_smiles[df_smiles['task'] == task]['auc_roc_delong'].values[0]
        smiles_lb = df_smiles[df_smiles['task'] == task]['auc_roc_delong_lb'].values[0]
        smiles_ub = df_smiles[df_smiles['task'] == task]['auc_roc_delong_ub'].values[0]
        smiles_error = [[smiles_auc - smiles_lb], [smiles_ub - smiles_auc]]

        # Get AUC-ROC and confidence intervals for SELFIES
        selfies_auc = df_selfies[df_selfies['task'] == task]['auc_roc_delong'].values[0]
        selfies_lb = df_selfies[df_selfies['task'] == task]['auc_roc_delong_lb'].values[0]
        selfies_ub = df_selfies[df_selfies['task'] == task]['auc_roc_delong_ub'].values[0]
        selfies_error = [[selfies_auc - selfies_lb], [selfies_ub - selfies_auc]]

        # Plot the bars for SMILES and SELFIES with error bars
        ax.bar(i - bar_width/2, smiles_auc, bar_width, label='SMILES' if i == 0 else "",
               yerr=smiles_error, error_kw=error_kw, color='blue', alpha=0.6)
        ax.bar(i + bar_width/2, selfies_auc, bar_width, label='SELFIES' if i == 0 else "",
               yerr=selfies_error, error_kw=error_kw, color='orange', alpha=0.6)

    # Labeling
    ax.set_xlabel('Tasks', fontweight='bold', fontsize=14)
    ax.set_ylabel('AUC-ROC', fontweight='bold', fontsize=14)
    ax.set_title('Comparison between SELFIES and SMILES Representations Across Different Tox21 Bioactivity Endpoints', fontsize=14)
    ax.set_xticks(index)
    ax.set_xticklabels(tasks, rotation=45, fontsize=12, fontweight='bold')
    ax.legend()

    # Show plot
    plt.tight_layout()

    # save plot
    logger.info(f"save plot at: ./plots/scores_performance/SMILES_VS_SELFIES_ENDPOINTS.png")
    plt.savefig(f'./plots/scores_performance/SMILES_VS_SELFIES_ENDPOINTS.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pre-train chemBERTa')
    parser.add_argument('--vocab_sizes', nargs='+', type=int, default=[100,200,500,1000,2000,5000,10000,20000,50000,100000,200000,500000], help='vocab sizes')
    parser.add_argument('--pretrained_data', type=str, default='smilesDB', help='pretrained dataset')
    parser.add_argument('--test_data', type=str, default='smilesTox21', help='test dataset')
    parser.add_argument('--target_data',nargs='+', type=str, default='tox21', help='target dataset')
    parser.add_argument('--mol_rep' , type=str, default='smiles', help='molecular representation')
    parser.add_argument('--tokenizer_data', type=str, default='smilesDB', help='tokenizer data')
    parser.add_argument('--tokenizer_type', type=str, default='BPE', help='tokenizer type')
    args = parser.parse_args()
    
    # for target_data in args.target_data:
    #     args.target_data = target_data

    #     for vocab_size in args.vocab_sizes:
    #         eval(args, vocab_size)

    
    # plot perplexity vs vocab size
    # csv_path = './logs/perplexity/perplexity_per_word_tox21.csv'
    # plot_perplexity_vs_vocab(csv_path, test_data='tox21')

    # csv_path = './logs/perplexity/perplexity_per_word_clintox.csv'
    # plot_perplexity_vs_vocab(csv_path, test_data='clintox')

    # csv_path = './logs/perplexity/perplexity_per_word_hips.csv'
    # plot_perplexity_vs_vocab(csv_path, test_data='hips')


    csv= './logs/end_points_fine_tune/SMILES_VS_SELFIES_ENDPOINTS_epochs20.csv'
    plot_auc_comparison(csv)
    






    



   

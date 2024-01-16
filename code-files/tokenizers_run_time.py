from toxic_tokenizer import Toxic_Tokenizer
import logging
import pandas as pd
import numpy as np
import argparse
import time
from utils import count_tokens
import matplotlib.pyplot as plt
from tokenization_algorithms.MacFragger import MacFragGenerator

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




def macfragger_decompose(train_data, max_blocks=6, max_sr=8, min_frag_atoms=1, as_mols=False):
    logger.info(f'MacFragger Training and Decomposition in progress for {train_data} dataset...')
    out_path = f'./datasets/pre_processed/{train_data}'
    input_file = f'./datasets/pre_processed/{train_data}.smi'

    macFragger = MacFragGenerator()
    macFragger.write_file(input_file, out_path, max_blocks, max_sr, as_mols, min_frag_atoms)

    # Count number of tokens
    tokens, unique_tokens = count_tokens(out_path + '_mac_fragments')
    logger.info(f'Number of tokens: {tokens}, Number of unique tokens: {unique_tokens}')
    logger.info(f'MacFragger decomposition on {train_data} dataset is done!')

# write function to compare between tokenizers based on run time
def  compare_run_time(tokenizer,smiles , vocab_size):
    # write function to compare between tokenizers based on run time
    start = time.time()
    # if tokenizer is MacFragger
    if tokenizer.tokenizer_type == 'MacFrag':
        macfragger_decompose('smilesTox21')

    for smile in smiles:
        tokenizer.tokenize(smile)
    end = time.time()
    print(f'run time for {tokenizer.tokenizer_type} - {vocab_size} is {end-start}')
    duration = end-start 
    return (vocab_size, duration)

def plot_vocab_vs_duration(tokenizer_performance):
    # Dictionary to hold plotting data
    plot_data = {}

    # Process data for plotting
    for (tokenizer_type, vocab_size), duration in tokenizer_performance.items():
        plot_data.setdefault(tokenizer_type, {'vocab_sizes': [], 'durations': []})
        plot_data[tokenizer_type]['vocab_sizes'].append(vocab_size)
        plot_data[tokenizer_type]['durations'].append(duration)

    # Create the plot with a specified figure size
    plt.figure(figsize=(12, 8))

    # Prepare to plot MacFrag and Atom-wise as horizontal lines
    macfrag_duration = np.mean(plot_data.get('MacFrag', {}).get('durations', []))
    atomwise_duration = np.mean(plot_data.get('Atom-wise', {}).get('durations', []))
    plt.axhline(y=macfrag_duration, color='r', linestyle='--', label='MacFrag')
    plt.axhline(y=atomwise_duration, color='g', linestyle='--', label='Atom-wise')
  
    # remove MacFrage and Atom-wise from plot_data
    plot_data.pop('MacFrag', None)
    plot_data.pop('Atom-wise', None)

    # Plot each tokenizer's performance
    for tokenizer_type, data in plot_data.items():
        # Sort by vocab size for a cleaner line plot
        vocab_sizes, durations = zip(*sorted(zip(data['vocab_sizes'], data['durations'])))
        
        # Plot line for each tokenizer type
        plt.plot(vocab_sizes, durations, label=f"{tokenizer_type}", marker='o')
  

        # # annotate y axis with run time values for each tokenizer
        # for i, txt in enumerate(durations):
        #     plt.annotate(f'{txt:.4f}', (vocab_sizes[i], durations[i]), fontsize=8)
        

    # Log scale for x and y axes to handle wide range of values
    plt.xscale('log')
    plt.yscale('log')

    

    # Label the axes and title the plot
    plt.xlabel('Vocabulary Size (log scale)', fontsize=12, fontweight='bold')
    plt.ylabel('Run Time (log scale) seconds', fontsize=12, fontweight='bold')
    plt.title('Tokenizers Run Time Across Different Vocabulary Sizes', fontsize=14)

    # Add grid, legend, and layout tightening
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()

    # Save the plot with high resolution
    plot_path = './plots/run_time/tokenizer_run_time.png'
    plt.savefig(plot_path, dpi=300)
    print(f'Plot saved to {plot_path}')
    # Close the plot to free up memory
    plt.close()



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

    
    # load atom_wise tokenizer
    # args.tokenizer_type = 'Atom-wise'
    # args.tokenizer_data = 'smilesDB'
    # vocab_size = 163
    # atomwise_tokenizer , vocab_size = get_tokenizer(args, vocab_size)

    # # load MacFrag tokenizer
    # args.tokenizer_type = 'MacFrag'
    # args.tokenizer_data = 'smilesDB'
    # vocab_size = 2493
    # MacFrag_tokenizer , vocab_size = get_tokenizer(args, vocab_size)
    
    # load data
    train_df, valid_df, test_df = load_dataframes(args.target_data)
    tox21_df = pd.concat([train_df, valid_df, test_df])
    smiles = tox21_df['smiles'].tolist()
    
    sub_word_tokenizers = { 'Atom-wise': [163],
                           'MacFrag':[2493],
                           'WordPiece':[100, 200, 500, 1000, 2000, 5000, 10000, 20000,50000, 100000, 200000, 500000],
                            'BPE': [100, 200, 500, 1000, 2000, 5000, 10000, 20000,50000, 100000, 200000, 500000], 
                            'SPE': [100, 200, 500, 1000, 2000, 5000, 10000, 20000,50000, 100000, 200000, 500000],
                            'Morfessor': [100, 200, 500, 1000, 2000, 5000, 10000, 20000,50000, 100000, 200000, 500000]}
    
    #100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000
    vocabs = []
    args.tokenizer_data = 'smilesDB'
    tokenizer_type_perofrmance = {}
    for tokenizer_type, vocab_sizes in sub_word_tokenizers.items():
        for vocab_size in vocab_sizes:
            args.tokenizer_type = tokenizer_type
            if args.tokenizer_type == 'Atom-wise':
                vocab_size = 163
            elif args.tokenizer_type == 'MacFrag':
                vocab_size = 2493
                
            # load sub-word tokenizers
            sub_word_tokenizer , vocab_size = get_tokenizer(args, vocab_size)
            vocab_size, duration = compare_run_time(sub_word_tokenizer,smiles , vocab_size)
            tokenizer_type_perofrmance[(tokenizer_type, vocab_size)] = duration


    logger.info(f'run time for {tokenizer_type_perofrmance}')
    plot_vocab_vs_duration(tokenizer_type_perofrmance)

    


    
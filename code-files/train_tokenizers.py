import argparse
import logging
import pandas as pd
import selfies as sf
from data_preprocess import DataPreprocessor
from utils import (
    read_segmented_mols, smiles_to_text, selfies_to_text,
    smiles_fragments_to_selfies_fragments, build_vocab,
    is_supported_smiles, count_tokens, save_model, load_model, preprocess_and_save_scaffold_splits
)
from tokenization_algorithms.MacFragger import MacFragGenerator
from tokenization_algorithms.bpe import BPELearner
from tokenization_algorithms.wordpiece import WordPieceTrainer
import codecs
from SmilesPE.tokenizer import SPE_Tokenizer
from SmilesPE.pretokenizer import atomwise_tokenizer , kmer_tokenizer
import pickle
import os
# Set up logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)



# Constants
SPECIAL_TOKENS = ['[PAD]', '[unused1]', '[unused2]', '[unused3]', '[unused4]', '[unused5]', '[unused6]',
                  '[unused7]', '[unused8]', '[unused9]', '[unused10]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']

# atom-level tokens used for trained the spe vocabulary
ATOM_TOKENS = ['[c-]', '[SeH]', '[N]', '[C@@]', '[Te]', '[OH+]', 'n', '[AsH]', '[B]', 'b', '[S@@]', 'o', ')', '[NH+]',
               '[SH]', 'O', 'I', '[C@]', '-', '[As+]', '[Cl+2]', '[P+]', '[o+]', '[C]', '[C@H]', '[CH2]', '\\', 'P',
               '[O-]', '[NH-]', '[S@@+]', '[te]', '[s+]', 's', '[B-]', 'B', 'F', '=', '[te+]', '[H]', '[C@@H]',
               '[Na]', '[Si]', '[CH2-]', '[S@+]', 'C', '[se+]', '[cH-]', '6', 'N', '[IH2]', '[As]', '[Si@]',
               '[BH3-]', '[Se]', 'Br', '[C+]', '[I+3]', '[b-]', '[P@+]', '[SH2]', '[I+2]', '%11', '[Ag-3]',
               '[O]', '9', 'c', '[N-]', '[BH-]', '4', '[N@+]', '[SiH]', '[Cl+3]', '#', '(', '[O+]', '[S-]',
               '[Br+2]', '[nH]', '[N+]', '[n-]', '3', '[Se+]', '[P@@]', '[Zn]', '2', '[NH2+]', '%10', '[SiH2]',
               '[nH+]', '[Si@@]', '[P@@+]', '/', '1', '[c+]', '[S@]', '[S+]', '[SH+]', '[B@@-]', '8', '[B@-]',
               '[C-]', '7', '[P@]', '[se]', 'S', '[n+]', '[PH]', '[I+]', '5', 'p', '[BH2-]', '[N@@+]', '[CH]',
               'Cl'] 




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


def train_wordpiece_tokenizer(train_data, vocab_size , min_frequency):
    logger.info(f'Training WordPiece encoders with vocabulary size {str(vocab_size)} for {train_data} dataset...')
    input_file = f'./datasets/pre_processed/{train_data}'
    # Train WordPiece tokenizer
    wordpiece_tokenizer = WordPieceTrainer(vocab_size=vocab_size, min_frequency= min_frequency)
    wordpiece_tok, vocab = wordpiece_tokenizer.train(input_file)
    wordpiece_tokenizer_pth = f"./models/tokenizers/wordpiece/wordpiece_{train_data}_{str(vocab_size)}.bin"
    save_model(wordpiece_tok, wordpiece_tokenizer_pth)
    logger.info(f'Training WordPiece encoders for {train_data} dataset is done!')

    # Test WordPiece tokenizer
    # Test the tokenizer
    if train_data == 'smilesDB':
        smi = 'COC(=O)c1c(N2C(=O)[C@@H]3[C@@H]4CCC[NH+]4[C@@]4(C(=O)Nc5c(C)cc(C)cc54)[C@@H]3C2=O)sc(C)c1C'
        logger.info(f'Original SMILES: {smi} ')
        logger.info(f'Encoded SMILES: {wordpiece_tok.encode(smi).tokens} on {train_data} dataset and vocabulary size {str(vocab_size)}')
    if train_data == 'selfiesDB':
        self = '[C][#Branch1][C][#N][Fe-2][Branch1][Ring1][C][#N][Branch1][Ring1][C][#N][Branch1][Ring1][C][#N][Branch1][Ring1][C][#N][N][=O]'
        logger.info(f'Original SMILES: {self} ')
        logger.info(f'Encoded SMILES: {wordpiece_tok.encode(self).tokens} on {train_data} dataset and vocabulary size {str(vocab_size)}')
        

def train_spe_tokenizer(train_data , vocab_size , min_frequency):
    logger.info(f'Training SPE encoders with vocabulary size {str(vocab_size)} for {train_data} dataset...')
    input_file = f'./datasets/pre_processed/{train_data}'
    output_file = f'./vocabs/spe_vocab/{train_data}_{str(vocab_size)}'
    # Train SPE tokenizer
    bpe_learner = BPELearner(file_name=input_file, output_file=output_file,num_symbols= vocab_size, min_frequency=min_frequency, augmentation=1, verbose=True, total_symbols=True)
    spe_tok, vocab = bpe_learner.learn_SMILES()
    logger.info(f'Training SPE encoders for {train_data} dataset is done!')

    # TEST SPE TOKENIZER
    spe_vob= codecs.open(output_file)
    spe = SPE_Tokenizer(spe_vob)
    if train_data == 'smilesDB':
        smi = 'COC(=O)c1c(N2C(=O)[C@@H]3[C@@H]4CCC[NH+]4[C@@]4(C(=O)Nc5c(C)cc(C)cc54)[C@@H]3C2=O)sc(C)c1C'
        logger.info(f'Original SMILE: {smi} ')
        logger.info(f'Encoded SMILE: {spe.tokenize(smi)} on {train_data} dataset and vocabulary size {str(vocab_size)}')

def train_bpe_tokenizer(train_data , vocab_size, min_frequency):
    logger.info(f'Training BPE encoders with vocabulary size {str(vocab_size)} for {train_data} dataset...')
    input_file = f'./datasets/pre_processed/{train_data}'
    # Train BPE tokenizer
    bpe_learner = BPELearner(file_name=input_file, output_file=None,num_symbols= vocab_size, min_frequency=min_frequency, augmentation=1, verbose=True, total_symbols=True)
    bpe_tok, vocab = bpe_learner.learn_BPE()
    bpe_tokenizer_pth = f"./models/tokenizers/bpe/{train_data}_{str(vocab_size)}.bin"
    save_model(bpe_tok, bpe_tokenizer_pth)
    logger.info(f'Training BPE encoders for {train_data} dataset is done!')

    # Test the tokenizer
    if train_data == 'smilesDB':
        smi = 'COC(=O)c1c(N2C(=O)[C@@H]3[C@@H]4CCC[NH+]4[C@@]4(C(=O)Nc5c(C)cc(C)cc54)[C@@H]3C2=O)sc(C)c1C'
        logger.info(f'Original SMILES: {smi} ')
        logger.info(f'Encoded SMILES: {bpe_tok.encode(smi).tokens} on {train_data} dataset and vocabulary size {str(vocab_size)}')
    if train_data == 'selfiesDB':
        self = '[C][#Branch1][C][#N][Fe-2][Branch1][Ring1][C][#N][Branch1][Ring1][C][#N][Branch1][Ring1][C][#N][Branch1][Ring1][C][#N][N][=O]'
        logger.info(f'Original SMILE: {self} ')
        logger.info(f'Encoded SMILE: {bpe_tok.encode(self).tokens} on {train_data} dataset and vocabulary size {str(vocab_size)}')


def build_vocab_for_atomwise_tokenizer(train_data , ATOM_TOKENS, SPECIAL_TOKENS):
    logger.info(f'Building vocabulary for atomwise tokenizer...')
    # Build vocabulary for atomwise tokenizer
    # Default vocabulary
    ATOM_CHARS = sorted(list(set(ATOM_TOKENS)))
    unique_tokens = SPECIAL_TOKENS + ATOM_CHARS
    unique_tokens = list(dict.fromkeys(unique_tokens))
    # remove space tokens isspace()
    unique_tokens = [token for token in unique_tokens if token and not token.isspace()]
    logger.info(f'Number of tokens (Atom-wise vocab): {len(unique_tokens)}')
    # Save the vocabulary
    Atom_vocab_path = f'./vocabs/atom_vocab/vocab_atomSMILES_{str(len(unique_tokens))}.txt' 
    with open(f'{Atom_vocab_path}', 'w') as f:
        for voc in unique_tokens:
            f.write(f'{voc}\n')
    
    # Build vocabulary from dataset
    # read file and build tokenizer
    input_file = f'./datasets/pre_processed/{train_data}'
    with open(input_file, 'r') as f:
        mols = f.readlines()
    mols = [mol.strip() for mol in mols]

    # split smiles with atom-tokenizer and build vocabulary atomwise_tokenizer(smi)
    logger.info(f'number of SMILES processed by atomwise tokenizer: {len(mols)}')
    uniqueChars = set()
    for mol in mols:
        uniqueChars.update(atomwise_tokenizer(mol))
        # uniqueChars.update(kmer_tokenizer(mol, ngram=1))
            
    uniqueChars = sorted(list(uniqueChars))
    uniqueChars = SPECIAL_TOKENS + uniqueChars
    uniqueChars = list(dict.fromkeys(uniqueChars))
    uniqueChars = [token for token in uniqueChars if token and not token.isspace()]
    logger.info(f'Number of tokens (Atom-wise vocab): {len(uniqueChars)}')
    # Save the vocabulary
    Atom_vocab_path = f'./vocabs/atom_vocab/{train_data}_{str(len(uniqueChars))}.txt'
    with open(f'{Atom_vocab_path}', 'w') as f:
        for voc in uniqueChars:
            f.write(f'{voc}\n')
    
    logger.info(f'Vocabulary for atomwise tokenizer is done!')

    # TEST ATOMWISE TOKENIZER
    if train_data == 'smilesDB' or train_data == 'smilesZinc':
        smi = 'COC(=O)c1c(N2C(=O)[C@@H]3[C@@H]4CCC[NH+]4[C@@]4(C(=O)Nc5c(C)cc(C)cc54)[C@@H]3C2=O)sc(C)c1C'
        logger.info(f'Original SMILE: {smi} ')
        logger.info(f'Encoded SMILE: {atomwise_tokenizer(smi)} with vocabulary size {str(len(unique_tokens))}')
    if train_data == 'selfiesDB' or train_data == 'selfiesZinc':
        self = '[C][#Branch1][C][#N][Fe-2][Branch1][Ring1][C][#N][Branch1][Ring1][C][#N][Branch1][Ring1][C][#N][Branch1][Ring1][C][#N][N][=O]'
        logger.info(f'Original SMILE: {self} ')
        logger.info(f'Encoded SMILE: {atomwise_tokenizer(self)} with vocabulary size {str(len(unique_tokens))}')

def build_vocab_for_macfragger(train_data , SPECIAL_TOKENS):
    # Build vocabulary for Mac Fragger fragments
    logger.info(f'Building vocabulary for MacFragger fragments on {train_data} dataset...')
    # input_file = f'./datasets/mac_fragments/{train_data}_fragments.smi'
    input_file = f'./datasets/pre_processed/{train_data}_mac_fragments'
    output_file = f'./vocabs/macFrag_vocab/{train_data}_vocab.smi'
    build_vocab(input_file, output_file, special_tokens=SPECIAL_TOKENS)
    logger.info(f'Vocabulary for MacFragger fragments on {train_data} dataset is done!')


def build_vocab_for_spe(train_data, vocab_size , SPECIAL_TOKENS , ATOM_TOKENS):
    logger.info(f'Building vocabulary for spe tokenizer On {train_data} with vocabulary size {str(vocab_size)}...')
    # Build vocabulary for spe tokenizer
    # './vocabs/spe_vocab/cleaned_smilesDB' + '_' + str(vocab_size) + '.smi', "r") as ins:
    with open(f'./vocabs/spe_vocab/{train_data}_{str(vocab_size)}', "r") as ins:
        spe_toks = []
        for line in ins:
            spe_toks.append(line.split('\n')[0])
    spe_tokens = []
    for s in spe_toks:
        spe_tokens.append(''.join(s.split(' ')))
    print('Number of TOKENS Pairs:', len(spe_toks))
    # build the vocabulary for the spe tokenizer
    # sort Atom tokens and spec tokens
    ATOM_CHARS = sorted(list(set(ATOM_TOKENS)))
    spe_tokens = sorted(list(set(spe_tokens)))
    spe_vocab = SPECIAL_TOKENS + ATOM_CHARS + spe_tokens
    spe_vocab = list(dict.fromkeys(spe_vocab))
    # remove space tokens isspace()
    spe_vocab = [token for token in spe_vocab if token and not token.isspace()]
    print('Number of tokens:', len(spe_vocab))
    # save the vocabulary
    with open(f'./vocabs/spe_vocab/{train_data}_{str(vocab_size)}.txt', 'w') as f:
        for voc in spe_vocab:
            f.write(f'{voc}\n')
    logger.info(f'Vocabulary for spe tokenizer is done!')

def build_vocab_for_bpe(train_data, vocab_size , SPECIAL_TOKENS):
    logger.info(f'Building vocabulary for bpe tokenizer On {train_data} with vocabulary size {str(vocab_size)}...')
    # Build vocabulary for bpe tokenizer
    bpe_tokenizer_pth = f"./models/tokenizers/bpe/{train_data}_{str(vocab_size)}.bin"
    bpe_tok = load_model(bpe_tokenizer_pth)
    unique_tokens = list(bpe_tok.get_vocab().keys())
    # sort unique tokens
    unique_tokens= sorted(list(set(unique_tokens)))
    vocs = SPECIAL_TOKENS + unique_tokens
    vocs = list(dict.fromkeys(vocs))
    # remove space tokens 
    vocs = [token for token in vocs if token and not token.isspace()]
    # vocab_path + 'bpe_DBsmi_vocab_' + str(vocab_size) + '.txt', 'w') as f:
    with open(f'./vocabs/bpe_vocab/{train_data}_{str(vocab_size)}.txt', 'w') as f:
        for voc in vocs:
            f.write(f'{voc.strip()}\n')
    logger.info(f'Vocabulary for bpe tokenizer is done!')

def build_vocab_for_wordpiece(train_data, vocab_size , SPECIAL_TOKENS):
    logger.info(f'Building vocabulary for wordpiece tokenizer On {train_data} with vocabulary size {str(vocab_size)}...')
    # Build vocabulary for wordpiece tokenizer
    wordpiece_tokenizer_pth = f"./models/tokenizers/wordpiece/wordpiece_{train_data}_{str(vocab_size)}.bin"
    wordpiece_tok = load_model(wordpiece_tokenizer_pth)
    unique_tokens = list(wordpiece_tok.get_vocab().keys())
    # sort unique tokens
    unique_tokens= sorted(list(set(unique_tokens)))
    vocs = SPECIAL_TOKENS + unique_tokens
    # remove duplicate ['UNK'] without sorting the list
    vocs = list(dict.fromkeys(vocs))
    # remove space tokens 
    vocs = [token for token in vocs if token and not token.isspace()]

    # remove empty tokens
    # vocs = list(filter(None, vocs))
    # Remove redundant tokens
    with open(f'./vocabs/wordpiece_vocab/{train_data}_{str(vocab_size)}.txt', 'w') as f:
        for voc in vocs:
            f.write(f'{voc.strip()}\n')
    logger.info(f'Vocabulary for wordpiece tokenizer is done!')

def build_vocab_for_morf(train_data, vocab_size , SPECIAL_TOKENS):
    # build vocab for morfessor tokenizers for smiles
    logger.info(f'Building vocabulary for morfessor tokenizers On {train_data} with vocabulary size {str(vocab_size)}...')
    morfessors_model_pt = f"./models/tokenizers/morfessors/morf_{train_data}_{str(vocab_size)}.bin"
    morfessors_vocab_path = f"./vocabs/morf_vocab/morf_{train_data}_{str(vocab_size)}.txt"
    logger.info(f'Loading morfessor model from {morfessors_model_pt}...')
    logger.info(f'Writing morfessor vocab to {morfessors_vocab_path}...')
    chem_morfessor= pickle.load(open(morfessors_model_pt, 'rb'))
    logger.info(f'Number of constructions: {len(chem_morfessor.get_constructions())}')
    constructions  = list(map(lambda t: t[0], chem_morfessor.get_constructions()))
    logger.info(f'Number of tokens: {len(constructions)}')
    # buil morfessor vocab file
    constructions = sorted(list(set(constructions)))
    vocs = SPECIAL_TOKENS + constructions
    # remove duplicate ['UNK'] without sorting the list
    vocs = list(dict.fromkeys(vocs))
    vocs = [token for token in vocs if token and not token.isspace()]
    with open(morfessors_vocab_path, 'w') as f:
        for voc in vocs:
            f.write(f'{voc}\n')
    
    # Test the morfessor tokenizer
    if train_data == 'smilesDB':
        smi = 'COC(=O)c1c(N2C(=O)[C@@H]3[C@@H]4CCC[NH+]4[C@@]4(C(=O)Nc5c(C)cc(C)cc54)[C@@H]3C2=O)sc(C)c1C'
        # logg molecule and tokens 
        logger.info(f'SMILE: {smi}')
        logger.info(f'Encoded SMILE: {chem_morfessor.viterbi_segment(smi)} on {train_data} with vocabulary size {str(vocab_size)}...')
    
    if train_data == 'selfiesDB':
        self = self = '[C][#Branch1][C][#N][Fe-2][Branch1][Ring1][C][#N][Branch1][Ring1][C][#N][Branch1][Ring1][C][#N][Branch1][Ring1][C][#N][N][=O]'
        logger.info(f'SMILE: {self}')
        logger.info(f'Encoded SMILE: {chem_morfessor.viterbi_segment(self)} on {train_data} with vocabulary size {str(vocab_size)}...')
    


def main(args , vocab_size):
    vocab_size = vocab_size
    train_data = args.train_data
    task = args.task
    

    if task == 'build_atomic_vocab':
        # build vocab for atomwise tokenizer
        build_vocab_for_atomwise_tokenizer(train_data, ATOM_TOKENS, SPECIAL_TOKENS)
    

    if task == 'train_MacFrag':
        macfragger_decompose(train_data , max_blocks=6, max_sr=4, min_frag_atoms=1, as_mols=False)
        build_vocab_for_macfragger(train_data, SPECIAL_TOKENS=SPECIAL_TOKENS)

   
    if task == 'train_wordpiece':
        # train wordpiece tokenizer
        train_wordpiece_tokenizer(train_data, vocab_size=vocab_size , min_frequency=1)
        build_vocab_for_wordpiece(train_data, vocab_size=vocab_size , SPECIAL_TOKENS=SPECIAL_TOKENS)
    
    if task == 'train_spe':
        # train spe tokenizer
        train_spe_tokenizer(train_data , vocab_size = vocab_size , min_frequency=1)
        build_vocab_for_spe(train_data , vocab_size = vocab_size, SPECIAL_TOKENS=SPECIAL_TOKENS , ATOM_TOKENS=ATOM_TOKENS)

    if task == 'train_bpe':
        # train bpe tokenizer
        train_bpe_tokenizer(train_data , vocab_size = vocab_size , min_frequency=1)
        build_vocab_for_bpe(train_data , vocab_size = vocab_size, SPECIAL_TOKENS=SPECIAL_TOKENS)
    
    if task == 'train_morf':
        # train morfessor tokenizer
        build_vocab_for_morf(train_data , vocab_size = vocab_size, SPECIAL_TOKENS=SPECIAL_TOKENS)
        

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Molecules Fragmentation')
    parser.add_argument('--task' , type=str, default=None, help='task name')
    parser.add_argument('--train_data' , type=str, default=None, help='path to train data')
    parser.add_argument('--vocab_sizes', nargs='+', type=int, default=None, help='vocab sizes')
    
    args =  parser.parse_args()
    
    if args.vocab_sizes is not None:
        for vocab_size in args.vocab_sizes:
            main(args,vocab_size)
    else:
            main(args , None)
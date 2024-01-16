from rdkit import Chem
from rdkit.Chem import AllChem
from collections import defaultdict
import itertools
import pandas as pd
import logging
from utils import *
from toxic_tokenizer import Toxic_Tokenizer
import argparse
logger = logging.getLogger(__name__)


DATASET_PATHS = {
    'clintox': './datasets/scaffold_splits/clintox_{}.csv',
    'tox21': './datasets/scaffold_splits/tox21_{}.csv',
    'HIPS': './datasets/HIPS/hips_{}.csv'
}

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



#####################################################################################

def generate_structural_alerts(smiles_data, min_occurrences, min_size, max_size , ppv_threshold, labels, tokenizer, atom_tokenizer):
    """
    Generate structural alerts from a dataset of SMILES strings.

    Parameters:
    - smiles_data: List of SMILES strings.
    - labels: List of labels (0 or 1) corresponding to each SMILES string.

    Returns:
    - alerts: List of structural alerts (substructures).
    """
    
    # Filter out positive (toxic) molecules
    positive_mols = [smiles for i, smiles in enumerate(smiles_data) if labels[i] == 1]

    # Generate fragments for each molecule
    def fragment_molecule(mol):
        frags = tokenizer.tokenize(mol)
        return  frags


    all_fragments = []
    for mol in positive_mols:
        all_fragments.extend(fragment_molecule(mol))

    # Count occurrences of each fragment
    fragment_counts = defaultdict(int)
    for frag in all_fragments:
        fragment_counts[frag] += 1
    
  
    
    def calculate_ppv(fragment):
        true_positives =  fragment_counts[fragment]
        false_positives = sum(1 for smiles, label in zip(smiles_data, labels) if fragment in smiles and label == 0)

        if true_positives + false_positives == 0:
            return 0
        

        return true_positives / (true_positives + false_positives)

    for fragment, count in fragment_counts.items():
        ppv = calculate_ppv(fragment)
        atoms = atom_tokenizer.tokenize(fragment)
        # logger.info(f"{fragment} : {atoms}")
        # logger.info(f"{fragment} {ppv} {count} {len(atoms)}")



    alerts = []
    for fragment, count in fragment_counts.items():
        if count >= min_occurrences:
            atoms = atom_tokenizer.tokenize(fragment)
            if len(atoms) >= min_size and len(atoms) <= max_size:
                ppv = calculate_ppv(fragment)
                if ppv > ppv_threshold:
                    alerts.append(fragment)

    # calculate over all avg ppv accross all fragments
    ppv_sum = 0
    for fragment , count in fragment_counts.items():
        ppv_sum += calculate_ppv(fragment)                

    return alerts , positive_mols

def active_covergae_percentage(smiles_data, labels, alerts):
    """
    Calculate the percentage of active molecules covered by a set of structural alerts.

    Parameters:
    - smiles_data: List of SMILES strings.
    - labels: List of labels (0 or 1) corresponding to each SMILES string.
    - alerts: List of structural alerts (substructures).

    Returns:
    - coverage: Percentage of active molecules covered by the alerts.
    """
    # Filter out positive (toxic) molecules
    positive_mols = [smiles for i, smiles in enumerate(smiles_data) if labels[i] == 1]

    # Calculate coverage
    coverage = 0
    for mol in positive_mols:
        for alert in alerts:
            if alert in mol:
                coverage += 1
                break

    return coverage / len(positive_mols)

def explore_structural_alerts(dataset_name,  min_occurrences = 3 ,min_size = 2,
    max_size = 40,
    ppv_threshold = 0.5, 
    tokenizer = None, atom_tokenizer = None):


    # NAMES : clintox , tox21 , HIPS
    train, val, test = load_dataframes(dataset_name)
    df = pd.concat([train, val, test])
    
    columns_names = df.columns
    logger.info(f"columns names of {dataset_name} target dataset: {columns_names}")
    smiles_data = df['smiles'].to_list()
    
    labels = {}
    # extract labels columns from dataframe
    for column in columns_names:
        if column not in  ['smiles' ,'smiles_mac_frags' , 'selfies' ,'labels']:
            logger.info(f"Extracting {column} label")
            labels[column] = df[column].to_list()
            
    
    data_set_alerts = []
    toxicity_alerts_coverage = {}
    # generate structural alerts for each label
    for label_name, label in labels.items():
        logger.info(f"Generating structural alerts for {label_name} label")
        alerts , pos_mols = generate_structural_alerts(smiles_data, min_occurrences, min_size, max_size , ppv_threshold, label, tokenizer, atom_tokenizer)
        # Coverage Percentage
        coverage = active_covergae_percentage(smiles_data, label, alerts)
        logger.info(f"Number of alerts: {len(alerts)}, number of positive molecules: {len(pos_mols)}")     
        logger.info(f"Coverage Percentage for data set {dataset_name} and label {label_name} : {coverage}")        
        data_set_alerts.extend(alerts)
        toxicity_alerts_coverage[label_name] = (smiles_data,pos_mols,alerts,coverage)

        

    # remove duplicates
    logger.info(f"Number of alerts before removing duplicates: {len(data_set_alerts)}")
    data_set_alerts = list(set(data_set_alerts))
    logger.info(f"Number of filtered  alerts: {len(data_set_alerts)}")

    return data_set_alerts , toxicity_alerts_coverage
        

         
def load_dataframes(target_data):
    if target_data not in DATASET_PATHS:
        raise ValueError('Train dataset not found')
    train_df = pd.read_csv(DATASET_PATHS[target_data].format('train'))
    valid_df = pd.read_csv(DATASET_PATHS[target_data].format('val'))
    test_df = pd.read_csv(DATASET_PATHS[target_data].format('test'))
    logger.info(f"columns names of {target_data} target dataset: {train_df.columns}")
    return train_df, valid_df, test_df



if __name__ == "__main__":
    # Example Usage 
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Toxicity detection')
    parser.add_argument('--tokenizer_type', type=str, default='BPE', help='tokenizer type')
    parser.add_argument('--task', type=str, default='NR-AR-LBD', help='task')
    parser.add_argument('--end_points', nargs='+', default=['NR-AR-LBD'], help='end points')
    parser.add_argument('--mol_rep', type=str, default='smiles', help='molecule representation')
    parser.add_argument('--target_data', type=str, default='tox21', help='target data')
    parser.add_argument('--tokenizer_data', type=str, default='smilesDB', help='tokenizer data')
    parser.add_argument('--vocab_size', type=int, default=10000, help='vocab size')
    parser.add_argument('--vocab_sizes', nargs='+', default=[10000], help='vocab sizes') 
    args = parser.parse_args()
    

    
    
    # load Data Driven base tokenizer
    BPE_tok , vocab = get_tokenizer(args, args.vocab_size)
    atom_tokenizer= Toxic_Tokenizer(vocab_file=f'vocabs/atom_vocab/{args.tokenizer_data}_{163}.txt', tokenizer_type='Atom-wise')
    

    min_occurrences = 3
    min_size = 2
    max_size = 40
    ppv_threshold = 0.5
    logger.info(f"Exploring structural alerts for dataset: clintox")
    clintox_data_set_alerts,clintox_label_alerts_coverage = explore_structural_alerts('clintox', min_occurrences ,min_size , max_size , ppv_threshold, BPE_tok, atom_tokenizer)
    logger.info(f"Exploring structural alerts for dataset: tox21")
    tox21_data_set_alerts,tox21_label_alerts_coverage = explore_structural_alerts('tox21', min_occurrences, min_size, max_size , ppv_threshold, BPE_tok, atom_tokenizer)
    logger.info(f"Exploring structural alerts for dataset: HIPS")
    hips_data_set_alerts,hips_label_alerts_coverage = explore_structural_alerts('HIPS', min_occurrences, min_size, max_size , ppv_threshold, BPE_tok, atom_tokenizer)
    
    
    # Pedict Toxicity_TYPE vs Structural_Alerts    
    # CLINTOX clintox_label_alerts_coverage
    logger.info(f"CLINTOX TOX TYPE , # MOLS , # POS MOLS , # ALERTS , COVERAGE")
    for label_name, (smiles_data,pos_mols,alerts,coverage) in clintox_label_alerts_coverage.items():
        logger.info(f"{label_name} , {len(smiles_data)-len(pos_mols)} , {len(pos_mols)} , {len(alerts)} , {coverage}")

    # TOX21 tox21_label_alerts_coverage
    logger.info(f"TOX21 TOX TYPE, # MOLS , # POS MOLS , # ALERTS , COVERAGE")
    for label_name, (smiles_data,pos_mols,alerts,coverage) in tox21_label_alerts_coverage.items():
        logger.info(f"{label_name} , {len(smiles_data)-len(pos_mols)} , {len(pos_mols)} , {len(alerts)} , {coverage}")

    # HIPS hips_label_alerts_coverage
    logger.info(f"HIPS TOX TYPE, # MOLS , # POS MOLS , # ALERTS , COVERAGE")
    for label_name, (smiles_data,pos_mols,alerts,coverage) in hips_label_alerts_coverage.items():
        logger.info(f"{label_name} , {len(smiles_data)-len(pos_mols)} , {len(pos_mols)} , {len(alerts)} , {coverage}")
    
    
    
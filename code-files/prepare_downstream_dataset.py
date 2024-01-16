from scaffold_preprocess import Scaffoldprocessor
from utils import  smiles_to_text 
from utils import *
import logging
import pandas as pd
# set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_data_for_dataset(tasks, dataset_name, split='scaffold'):
    """Process data for a given dataset."""
    data_processor = Scaffoldprocessor(tasks_wanted=tasks, split=split)
    tasks, train_df, valid_df, test_df, transformers = data_processor.process_data(dataset_name)
    return train_df, valid_df, test_df

def add_mac_fragments_to_df(df, fragmented_mols_path):
    """Add mac fragments to the dataframe."""
    fragmented_mols_smi = read_segmented_mols(fragmented_mols_path)
    df_copy = df.copy()  # Create a copy to avoid modifying the original DataFrame
    df_copy['smiles_mac_frags'] = fragmented_mols_smi
    return df_copy

def convert_smiles_to_selfies_and_filter(df, columns):
    """Convert SMILES in DataFrame to SELFIES and filter columns."""
    df_copy = df[df['smiles'].apply(is_supported_smiles)].copy()  # Filter and create a copy
    df_copy.loc[:, 'selfies'] = df_copy['smiles'].apply(sf.encoder)
    return df_copy[columns]


def prepare_downstream_datasets():
    """Prepare downstream target datasets."""
    logger.info('Preparing downstream target datasets...')
    logger.info('Moving scaffolds splits into preprocessed datasets files for Macfrag training...')

    clin_train_df, clin_valid_df, clin_test_df = process_data_for_dataset(['FDA_APPROVED','CT_TOX'], "clintox")
    tox21_train_df, tox21_valid_df, tox21_test_df = process_data_for_dataset(['NR-AR', 'NR-AR-LBD', 'NR-AhR','NR-Aromatase', 
                                                                              'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 
                                                                              'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'], "tox21")

    # Add mac fragments to the dataframes
    clin_train_df = add_mac_fragments_to_df(clin_train_df, './datasets/pre_processed/clintox_train_mac_fragments')
    clin_valid_df = add_mac_fragments_to_df(clin_valid_df, './datasets/pre_processed/clintox_val_mac_fragments')
    clin_test_df = add_mac_fragments_to_df(clin_test_df, './datasets/pre_processed/clintox_test_mac_fragments')
    tox21_train_df = add_mac_fragments_to_df(tox21_train_df, './datasets/pre_processed/tox21_train_mac_fragments')
    tox21_valid_df = add_mac_fragments_to_df(tox21_valid_df, './datasets/pre_processed/tox21_val_mac_fragments')
    tox21_test_df = add_mac_fragments_to_df(tox21_test_df, './datasets/pre_processed/tox21_test_mac_fragments')

    # Convert SMILES to SELFIES and filter columns
    clin_columns = ['smiles','smiles_mac_frags','selfies','labels','CT_TOX','FDA_APPROVED']
    tox21_columns = ['smiles','smiles_mac_frags','selfies','labels','NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER','NR-ER-LBD','NR-PPAR-gamma','SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53']

    clin_train_df = convert_smiles_to_selfies_and_filter(clin_train_df, clin_columns)
    clin_valid_df = convert_smiles_to_selfies_and_filter(clin_valid_df, clin_columns)
    clin_test_df = convert_smiles_to_selfies_and_filter(clin_test_df, clin_columns)
    tox21_train_df = convert_smiles_to_selfies_and_filter(tox21_train_df, tox21_columns)
    tox21_valid_df = convert_smiles_to_selfies_and_filter(tox21_valid_df, tox21_columns)
    tox21_test_df = convert_smiles_to_selfies_and_filter(tox21_test_df, tox21_columns)

    # Log shapes of the dataframes
    logger.info(f'ClinTox train shape: {clin_train_df.shape}')
    logger.info(f'ClinTox valid shape: {clin_valid_df.shape}')
    logger.info(f'ClinTox test shape: {clin_test_df.shape}')
    logger.info(f'Tox21 train shape: {tox21_train_df.shape}')
    logger.info(f'Tox21 valid shape: {tox21_valid_df.shape}')
    logger.info(f'Tox21 test shape: {tox21_test_df.shape}')

    # Save the dataframes
    clin_train_df.to_csv('./datasets/scaffold_splits/clintox_train.csv', index=False)
    clin_valid_df.to_csv('./datasets/scaffold_splits/clintox_val.csv', index=False)
    clin_test_df.to_csv('./datasets/scaffold_splits/clintox_test.csv', index=False)
    tox21_train_df.to_csv('./datasets/scaffold_splits/tox21_train.csv', index=False)
    tox21_valid_df.to_csv('./datasets/scaffold_splits/tox21_val.csv', index=False)
    tox21_test_df.to_csv('./datasets/scaffold_splits/tox21_test.csv', index=False)

    print('Done!')

def log_samples(dataset_name, train_df, valid_df, test_df):
    """Logs the first 5 samples of smiles and their corresponding mac fragments."""
    for split_name, df in {"train": train_df, "val": valid_df, "test": test_df}.items():
        logger.info(f'{dataset_name} {split_name} smiles: {df["smiles"].to_list()[0:1]}')
        logger.info(f'{dataset_name} {split_name} smiles mac fragments: {df["smiles_mac_frags"].to_list()[0:1]}')



if __name__ == "__main__":
    logger.info('Preparing downstream target datasets...')
    prepare_downstream_datasets()
    logger.info('Logging samples from both clintox and tox21 of smiles and corresponding smiles mac fragments...')
    clin_train_df = pd.read_csv('./datasets/scaffold_splits/clintox_train.csv')
    clin_valid_df = pd.read_csv('./datasets/scaffold_splits/clintox_val.csv')
    clin_test_df = pd.read_csv('./datasets/scaffold_splits/clintox_test.csv')
    tox21_train_df = pd.read_csv('./datasets/scaffold_splits/tox21_train.csv')
    tox21_valid_df = pd.read_csv('./datasets/scaffold_splits/tox21_val.csv')
    tox21_test_df = pd.read_csv('./datasets/scaffold_splits/tox21_test.csv')
    log_samples('clintox', clin_train_df, clin_valid_df, clin_test_df)
    log_samples('tox21', tox21_train_df, tox21_valid_df, tox21_test_df)
    
    
    
    
    # # read text file smiles and convert to selfies
    # with open('./datasets/pre_processed/smilesZinc', 'r') as f:
    #    smiles = f.readlines()
    # smiles = [smile.strip() for smile in smiles]
        
    # # convert smiles to selfies
    # selfies = []
    # for smile in smiles:
    #         if is_supported_smiles(smile):
    #             selfies.append(sf.encoder(smile))
    
    # # log number of supported smiles
    # logger.info(f'Number of supported smiles: {len(selfies)}')
    # # number of unsupported smiles
    # logger.info(f'Number of unsupported smiles: {len(smiles) - len(selfies)}')
    # # save selfies to text file
    # with open('./datasets/pre_processed/selfiesZinc', 'w') as f:
    #     for selfie in selfies:
    #         f.write("%s\n" % selfie)    
        

    # logger.info('Done!')





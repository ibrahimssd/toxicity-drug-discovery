import logging
import pandas as pd
import selfies as sf
from data_preprocess import DataPreprocessor
from utils import (
   smiles_to_text, selfies_to_text,
    is_supported_smiles, preprocess_and_save_scaffold_splits
)

# Set up logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


# Description: Preprocess datasets and save them in a format that can be used by the model
def preprocess_datasets():
    # Preprocess clintox dataset
    logger.info('Preprocessing clintox dataset...')
    clintox_preprocessor = DataPreprocessor(input_file='datasets/clintox.csv',
                                             output_path='datasets/pre_processed',
                                            dataset_name='clintox')
    clintox_data = clintox_preprocessor.pre_process()
    smiles_to_text(dataframe=clintox_data, output_file='datasets/pre_processed/smilesClintox.smi')
    smiles_to_text(dataframe=clintox_data, output_file='datasets/pre_processed/smilesClintox')
    selfies_to_text(dataframe=clintox_data, output_file='datasets/pre_processed/selfiesClintox')
    
    # Preprocess tox21 dataset
    logger.info('Preprocessing tox21 dataset...')
    tox21_preprocessor = DataPreprocessor(input_file='datasets/tox21.csv', 
                                          output_path='datasets/pre_processed',
                                            dataset_name='tox21')
    tox21_data = tox21_preprocessor.pre_process()
    
    smiles_to_text(dataframe=tox21_data, output_file='datasets/pre_processed/smilesTox21.smi')
    smiles_to_text(dataframe=tox21_data, output_file='datasets/pre_processed/smilesTox21')
    selfies_to_text(dataframe=tox21_data, output_file='datasets/pre_processed/selfiesTox21')
    
    # preprocess zinc dataset
    logger.info('Preprocessing zinc dataset...')
    zinc_preprocessor = DataPreprocessor(input_file='datasets/zinc.csv', 
                                          output_path='datasets/pre_processed',
                                            dataset_name='zinc')
    
    zinc_data = zinc_preprocessor.pre_process()

    smiles_to_text(dataframe=zinc_data, output_file='datasets/pre_processed/smilesZinc.smi')
    smiles_to_text(dataframe=zinc_data, output_file='datasets/pre_processed/smilesZinc')
    selfies_to_text(dataframe=zinc_data, output_file='datasets/pre_processed/selfiesZinc')
    
    # preprocess DB dataset
    logger.info('Preprocessing DB dataset...')
    DB_preprocessor = DataPreprocessor(input_file='datasets/DB.csv',
                                                output_path='datasets/pre_processed',
                                                dataset_name='DB')
    DB_data = DB_preprocessor.pre_process()
    smiles_to_text(dataframe=DB_data, output_file='datasets/pre_processed/smilesDB.smi')
    smiles_to_text(dataframe=DB_data, output_file='datasets/pre_processed/smilesDB')
    selfies_to_text(dataframe=DB_data, output_file='datasets/pre_processed/selfiesDB')

    
    # preprocess pubchem dataset
    # pubchem_preprocessor = DataPreprocessor(input_file='datasets/pubchem-500k.csv',
    #                                             output_path='datasets/pre_processed',
    #                                             dataset_name='pubchem500k')
    # pubchem_data = pubchem_preprocessor.pre_process()
    # smiles_to_text(dataframe=pubchem_data, output_file='datasets/pre_processed/smilesPubchem500k.smi')
    # smiles_to_text(dataframe=pubchem_data, output_file='datasets/pre_processed/smilesPubchem500k')
    # selfies_to_text(dataframe=pubchem_data, output_file='datasets/pre_processed/selfiesPubchem500k')


    # ... Preprocess other datasets if needed
    logger.info('Preprocessing scafold splits (train , val , test for clintox , tox21)...')
    preprocess_and_save_scaffold_splits()    


if __name__ == '__main__':
    preprocess_datasets()        
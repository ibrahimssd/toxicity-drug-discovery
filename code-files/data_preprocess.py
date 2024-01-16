import pandas as pd
import logging
from rdkit import Chem
import selfies as sf
import os

logging.basicConfig(level=logging.INFO)

class DataPreprocessor:
    def __init__(self, input_file, output_path, dataset_name):
        self.input_file = input_file
        self.output_path = output_path
        self.dataset_name = dataset_name

    @staticmethod
    def is_valid_smiles(smile):
        mol = Chem.MolFromSmiles(smile)
        return mol is not None

    @staticmethod
    def is_supported_smiles(smile):
        try:
            sf.encoder(smile)
            return True
        except Exception:
            logging.info("Unsupported SMILES: %s", smile)
            return False

    def pre_process(self):
        '''
        This function takes a csv file as input_file and returns a pre-processed csv file.
        The pre-processing steps are:
        1. Remove invalid SMILES
        2. Remove molecules containing * symbol
        3. Remove non-supported SMILES
        4. Convert SMILES to SELFIES
        5. Save to file
        '''
        print(f'Pre-processing {self.dataset_name} dataset...')
        df = pd.read_csv(self.input_file)
        df = df.drop_duplicates(subset=['smiles'])
        df = df.dropna()
        df = df[['smiles']]
        print(f'Number Of processed SMILES: {len(df)}')
        df_filtered = df[df['smiles'].apply(self.is_valid_smiles)]
        df_filtered = df_filtered[df_filtered['smiles'].apply(self.is_supported_smiles)]
        # remove duplicates 
        df_filtered = df_filtered.drop_duplicates(subset=['smiles'])
        print(f'Number of removed SMILES: {len(df) - len(df_filtered)}')
        df_filtered['selfies'] = df_filtered['smiles'].apply(sf.encoder)
        df_filtered = df_filtered.reset_index(drop=True)
        print(f'Number of valid SMILES: {len(df_filtered)}')
        output_filepath = os.path.join(self.output_path, f'{self.dataset_name}.csv')
        print(f'saving {self.dataset_name} dataset to CSV file...')
        df_filtered.to_csv(output_filepath, index=False)
        return df_filtered



import pandas as pd
import logging


def smiles_to_text(dataframe, output_file):
    # write smiles string representation in dataframe to text file
    with open(output_file, 'w') as f:
        for smile in dataframe['SMILES']:
            f.write("%s\n" % smile)


if __name__ == '__main__':
    # read data
    dataframe = pd.read_csv('data.csv')
    # filter nan values from SMILES colums 
    dataframe = dataframe[dataframe['SMILES'].notna()]
    # remove duplicated SMILES
    dataframe = dataframe.drop_duplicates(subset=['SMILES'])
    # convert SMILES to text file
    smiles_to_text(dataframe, 'chembl_smiles.txt')
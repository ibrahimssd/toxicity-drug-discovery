from rdkit import Chem
import selfies as sf
import logging
from selfies import DecoderError


def is_valid_smiles(smile):
        mol = Chem.MolFromSmiles(smile)
        return mol is not None
def is_supported_smiles(smile):
        try:
            sf.encoder(smile)
            return True
        except Exception:
            logging.info("Unsupported SMILES: %s", smile)
            return False
def clean_smiles(input_file, output_file):
    cleaned_smiles = []
    invalid_smiles = 0
    with open(input_file, 'r') as file:
        num_mols = 0
        for line in file:
            num_mols += 1
            smile = line.strip()
            # apply smiles.replace('\\\\', '\\') 
            smile = smile.replace('\\\\', '\\')
            # remove unrecognized symbol '\'
            smile = smile.replace('\\', '')
            # remove * symbol
            smile = smile.replace('*', '')
            if is_valid_smiles(smile) and is_supported_smiles(smile):
                cleaned_smiles.append(smile)
            
    
    print('Number of invalid SMILES:', num_mols - len(cleaned_smiles))
    print('Number of valid SMILES:', len(cleaned_smiles))
    # count files lines 
    with open(input_file, 'r') as file:
        num_lines = sum(1 for line in file)
    with open(output_file, 'w') as file:
        for smile in cleaned_smiles:
            file.write(smile + '\n')


if __name__ == '__main__':
    input_file = 'smilesDB.smi'
    output_file = 'cleaned_smilesDB.smi'
    clean_smiles(input_file, output_file)
    # CONVERT SMILES TO SELFIES
    file_name = 'cleaned_smilesDB.txt'
    with open(file_name, "r") as ins:
        SMILES = []
        for line in ins:
            SMILES.append(line.split('\n')[0])
    print('Number of SMILES:', len(SMILES))
    # convert SMILES to SELFIES
    SELFIES = []
    for smi in SMILES:
        try:
            selfie = sf.encoder(smi)
            SELFIES.append(selfie)
        except DecoderError:
            print('DecoderError: ', smi)
            continue
    print('Number of SELFIES:', len(SELFIES))
    # save SELFIES to file
    with open('./cleaned_selfiesDB.txt', 'w') as f:
        for selfie in SELFIES:
            f.write("%s\n" % selfie)
            
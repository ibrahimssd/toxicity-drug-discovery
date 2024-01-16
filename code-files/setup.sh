#!/bin/bash
# Create a new conda environment with Python 3.7
# conda create --name drug-discovery python=3.7
# Activate the new environment
# conda activate drug-discovery
# Install the necessary packages

conda install -c anaconda ipykernel
conda install -c conda-forge selfies
conda install pytorch torchvision torchaudio -c pytorch
conda install numpy
conda install pandas
conda install -c conda-forge matplotlib
conda install scikit-learn
conda install -c huggingface datasets
conda install -c rdkit rdkit
conda install -c conda-forge tqdm
conda install -c conda-forge python-igraph
conda install -c conda-forge tokenizers
conda install -c conda-forge morfessor
conda install -c "conda-forge/label/cf202003" transformers
conda install -c openbabel openbabel
conda install -c deepchem deepchem
conda install -c anaconda boto3
conda install -c dglteam dgl
pip install SmilesPE
conda install -c conda-forge dgllife
conda install -c anaconda tensorflow
conda install -c conda-forge pytorch-lightning
conda install -c conda-forge tensorboard
conda install -c conda-forge jax jaxlib
conda install -c conda-forge imbalanced-learn
# Install the necessary packages with pip
pip install --upgrade transformers
pip install accelerate >=0.20.1
pip install simpletransformers
pip install evaluate

# new packages
pip install transformers-interpret
pip install html2image
pip install imgkit
pip install selenium
sudo pip3 install IPython
pip install ipython
pip install nlp


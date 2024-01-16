# !/usr/bin/env bash
nvidia-smi
echo $CUDA_VISIBLE_DEVICES
echo $HOSTNAME
which python

# DATASET PREPARATION
# CUDA_LAUNCH_BLOCKING=1 python tokenizing_datasets_preprocessing.py

# DOWNSTREAM TASKS DATA PREPARATION
# DECOMPOSE CLINTOX TRAIN VALIDATION AND TEST SETS FOR DOWNSTREAM TASKS
# CUDA_LAUNCH_BLOCKING=1 python train_tokenizers.py  --task 'train_MacFrag'\
#                                                     --train_data 'clintox_train'\

# CUDA_LAUNCH_BLOCKING=1 python train_tokenizers.py  --task 'train_MacFrag'\
#                                                     --train_data 'clintox_val'\

# CUDA_LAUNCH_BLOCKING=1 python train_tokenizers.py  --task 'train_MacFrag'\
#                                                     --train_data 'clintox_test'\

# # DECOMPOSE TOX21 TRAIN VALIDATION AND TEST SETS FOR DOWNSTREAM TASKS
# CUDA_LAUNCH_BLOCKING=1 python train_tokenizers.py  --task 'train_MacFrag'\
#                                                     --train_data 'tox21_train'\

# CUDA_LAUNCH_BLOCKING=1 python train_tokenizers.py  --task 'train_MacFrag'\
#                                                     --train_data 'tox21_val'\

# CUDA_LAUNCH_BLOCKING=1 python train_tokenizers.py  --task 'train_MacFrag'\
#                                                     --train_data 'tox21_test'\


# TRAIN TOKENIZERS
# SMILES DB DATABASE
# ATOMIC TOKENIZERS
# SMILES
# CUDA_LAUNCH_BLOCKING=1 python train_tokenizers.py --task 'build_atomic_vocab'\
#                                                     --train_data 'smilesDB'\

# # SELFIES                                                    
# CUDA_LAUNCH_BLOCKING=1 python train_tokenizers.py --task 'build_atomic_vocab'\
#                                                     --train_data 'selfiesDB'\


# CHEMICAL-RULES BASED TOKENIZERS
# CUDA_LAUNCH_BLOCKING=1 python train_tokenizers.py  --task 'train_MacFrag'\
#                                                     --train_data 'smilesClintox'\
                                                 
# CUDA_LAUNCH_BLOCKING=1 python train_tokenizers.py  --task 'train_MacFrag'\
#                                                     --train_data 'smilesTox21'\

# CUDA_LAUNCH_BLOCKING=1 python train_tokenizers.py  --task 'train_MacFrag'\
#                                                     --train_data 'smilesZinc'\

# CUDA_LAUNCH_BLOCKING=1 python train_tokenizers.py  --task 'train_MacFrag'\
#                                                     --train_data 'smilesDB'\

# CUDA_LAUNCH_BLOCKING=1 python train_tokenizers.py  --task 'train_MacFrag'\
#                                                     --train_data 'smilesPubChem500k'\


# DATA-DRIVEN TOKENIZERS
# CUDA_LAUNCH_BLOCKING=1 python train_tokenizers.py  --task 'train_wordpiece'\
#                                                     --train_data 'smilesDB'\
#                                                     --vocab_sizes 100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000\


# CUDA_LAUNCH_BLOCKING=1 python train_tokenizers.py  --task 'train_spe'\
#                                                     --train_data 'smilesDB'\
#                                                     --vocab_sizes 100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000\

CUDA_LAUNCH_BLOCKING=1  python train_tokenizers.py   --task 'train_bpe'\
                                                     --train_data 'smilesDB'\
                                                     --vocab_sizes 100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000


# CUDA_LAUNCH_BLOCKING=1 python train_tokenizers.py  --task 'train_morf'\
#                                                     --train_data 'smilesDB'\
#                                                     --vocab_sizes 100000 200000 500000\




# SMILES ZINC DATABASE
# ATOMIC TOKENIZERS
# SMILES
# CUDA_LAUNCH_BLOCKING=1 python train_tokenizers.py --task 'build_atomic_vocab'\
#                                                     --train_data 'smilesZinc'\

# # SELFIES                                                    
# CUDA_LAUNCH_BLOCKING=1 python train_tokenizers.py --task 'build_atomic_vocab'\
#                                                     --train_data 'selfiesZinc'\




# # DATA-DRIVEN TOKENIZERS
# CUDA_LAUNCH_BLOCKING=1 python train_tokenizers.py  --task 'train_wordpiece'\
#                                                     --train_data 'smilesZinc'\
#                                                     --vocab_sizes 100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000\


# CUDA_LAUNCH_BLOCKING=1 python train_tokenizers.py  --task 'train_spe'\
#                                                     --train_data 'smilesZinc'\
#                                                     --vocab_sizes 100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000\

# CUDA_LAUNCH_BLOCKING=1  python train_tokenizers.py   --task 'train_bpe'\
#                                                      --train_data 'smilesZinc'\
#                                                      --vocab_sizes 100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000\


# CUDA_LAUNCH_BLOCKING=1 python train_tokenizers.py  --task 'train_morf'\
#                                                     --train_data 'smilesZinc'\
#                                                     --vocab_sizes  100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000\




# # CHEMICAL-RULES BASED TOKENIZERS
# CUDA_LAUNCH_BLOCKING=1 python train_tokenizers.py  --task 'train_MacFrag'\
#                                                     --train_data 'smilesClintox'\
                                                 
# CUDA_LAUNCH_BLOCKING=1 python train_tokenizers.py  --task 'train_MacFrag'\
#                                                     --train_data 'smilesTox21'\

# CUDA_LAUNCH_BLOCKING=1 python train_tokenizers.py  --task 'train_MacFrag'\
#                                                     --train_data 'smilesZinc'\

# CUDA_LAUNCH_BLOCKING=1 python train_tokenizers.py  --task 'train_MacFrag'\
#                                                     --train_data 'smilesDB'\

# CUDA_LAUNCH_BLOCKING=1 python train_tokenizers.py  --task 'train_MacFrag'\
#                                                     --train_data 'smilesPubChem500k'\


#!/usr/bin/env bash
# run misc. stuff
# nvidia-smi
# htop
#watch -n0.1 nvidia-smi&
#nvitop -m
#gpustat&
#nvtop
echo $CUDA_VISIBLE_DEVICES
echo $HOSTNAME
which python


# Path: fine_tune_best_tokenizer.sh
#'NR-AR', 'NR-AR-LBD','NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
#'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']

############## TEST ##############################
# CUDA_LAUNCH_BLOCKING=1 python fine_tune_best_tokenizer.py --vocab_size 50000\
#                                            --target_data tox21\
#                                            --mol_rep selfies\
#                                            --tasks SR-MMP SR-HSE SR-ATAD5 SR-ARE NR-PPAR-gamma NR-ER-LBD NR-ER NR-Aromatase NR-AhR NR-AR-LBD NR-AR SR-p53\
#                                             --tokenizer_type WordPiece\
#                                             --tokenizer_data smilesDB\
#                                            --pretrained_data smilesDB\
#                                            --epochs 1\
#                                            --aug


######################################################### SELFIES vs SMILES #####################################################################################################

CUDA_LAUNCH_BLOCKING=1 python fine_tune_best_tokenizer.py --vocab_size 10000\
                                           --target_data tox21\
                                           --mol_rep smiles\
                                           --tasks SR-MMP SR-HSE SR-ATAD5 SR-ARE NR-PPAR-gamma NR-ER-LBD NR-ER NR-Aromatase NR-AhR NR-AR-LBD NR-AR SR-p53\
                                            --tokenizer_type BPE\
                                            --tokenizer_data smilesDB\
                                           --pretrained_data smilesDB\
                                           --epochs 20\
                                           --aug

CUDA_LAUNCH_BLOCKING=1 python fine_tune_best_tokenizer.py --vocab_size 10000\
                                           --target_data tox21\
                                           --mol_rep selfies\
                                            --tasks SR-MMP SR-HSE SR-ATAD5 SR-ARE NR-PPAR-gamma NR-ER-LBD NR-ER NR-Aromatase NR-AhR NR-AR-LBD NR-AR SR-p53\
                                            --tokenizer_type BPE\
                                            --tokenizer_data selfiesDB\
                                           --pretrained_data selfiesDB\
                                           --epochs 20\
                                           --aug



################################################ Perplexity  Tox21 #############################################################################################
# CUDA_LAUNCH_BLOCKING=1 python fine_tune_best_tokenizer.py --vocab_size 163\
#                                            --target_data tox21\
#                                            --mol_rep smiles\
#                                            --task SR-p53\
#                                             --tokenizer_type Atom-wise\
#                                             --tokenizer_data smilesDB\
#                                            --pretrained_data smilesDB\
#                                            --epochs 30\
#                                            --aug

# CUDA_LAUNCH_BLOCKING=1 python fine_tune_best_tokenizer.py --vocab_size 204\
#                                            --target_data tox21\
#                                            --mol_rep selfies\
#                                             --task SR-p53\
#                                             --tokenizer_type Atom-wise\
#                                             --tokenizer_data selfiesDB\
#                                            --pretrained_data selfiesDB\
#                                            --epochs 30\
#                                            --aug


# CUDA_LAUNCH_BLOCKING=1 python fine_tune_best_tokenizer.py   --vocab_sizes 100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000\
#                                                             --tasks  SR-p53\
#                                                             --target_data tox21\
#                                                             --mol_rep smiles\
#                                                             --tokenizer_type BPE\
#                                                             --tokenizer_data smilesDB\
#                                                             --pretrained_data smilesDB\
#                                                             --epochs 30\
#                                                             --aug




################################# Perplexity CLINTOX #####################################################################################################



# CUDA_LAUNCH_BLOCKING=1 python fine_tune_best_tokenizer.py --vocab_size 163\
#                                            --target_data clintox\
#                                            --mol_rep smiles\
#                                            --task CT_TOX\
#                                             --tokenizer_type Atom-wise\
#                                             --tokenizer_data smilesDB\
#                                            --pretrained_data smilesDB\
#                                            --epochs 30\
#                                            --no-aug

# CUDA_LAUNCH_BLOCKING=1 python fine_tune_best_tokenizer.py  --vocab_size 204\
#                                            --target_data clintox\
#                                            --mol_rep selfies\
#                                             --task CT_TOX\
#                                             --tokenizer_type Atom-wise\
#                                             --tokenizer_data selfiesDB\
#                                            --pretrained_data selfiesDB\
#                                            --epochs 30\
#                                            --no-aug

# CUDA_LAUNCH_BLOCKING=1 python fine_tune_best_tokenizer.py   --vocab_sizes 100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000\
#                                                             --tasks  CT_TOX\
#                                                             --target_data clintox\
#                                                             --mol_rep smiles\
#                                                             --tokenizer_type BPE\
#                                                             --tokenizer_data smilesDB\
#                                                             --pretrained_data smilesDB\
#                                                             --epochs 30\
#                                                             --no-aug


################################################### Perplexity HIPS #####################################################################################################

# CUDA_LAUNCH_BLOCKING=1 python fine_tune_best_tokenizer.py --vocab_size 163\
#                                            --target_data hips\
#                                            --mol_rep smiles\
#                                            --task y\
#                                             --tokenizer_type Atom-wise\
#                                             --tokenizer_data smilesDB\
#                                            --pretrained_data smilesDB\
#                                            --epochs 30\
#                                            --no-aug


# CUDA_LAUNCH_BLOCKING=1 python fine_tune_best_tokenizer.py   --vocab_sizes 100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000\
#                                                             --tasks  y\
#                                                             --target_data hips\
#                                                             --mol_rep smiles\
#                                                             --tokenizer_type BPE\
#                                                             --tokenizer_data smilesDB\
#                                                             --pretrained_data smilesDB\
#                                                             --epochs 30\
#                                                             --no-aug




   

#############################################################################################################################################################

# CUDA_LAUNCH_BLOCKING=1 python fine_tune_best_tokenizer.py   --vocab_sizes 10000 20000 50000 100000 200000 500000\
#                                                             --tasks   SR-MMP SR-HSE SR-ATAD5 SR-ARE NR-PPAR-gamma NR-ER-LBD NR-ER NR-Aromatase NR-AhR NR-AR-LBD NR-AR SR-p53\
#                                                             --target_data tox21\
#                                                             --mol_rep smiles\
#                                                             --tokenizer_type BPE\
#                                                             --tokenizer_data smilesDB\
#                                                             --pretrained_data smilesDB\
#                                                             --epochs 10\
#                                                             --no-aug



# HIPS TRAINING
# CUDA_LAUNCH_BLOCKING=1 python fine_tune_best_tokenizer.py   --vocab_sizes 100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000\
#                                                             --tasks  y\
#                                                             --target_data hips\
#                                                             --mol_rep smiles\
#                                                             --tokenizer_type BPE\
#                                                             --tokenizer_data smilesDB\
#                                                             --pretrained_data smilesDB\
#                                                             --epochs 100\
#                                                             --no-aug
                                                            




# CUDA_LAUNCH_BLOCKING=1 python fine_tune_best_tokenizer.py   --vocab_sizes 5000 10000 20000 50000 100000 200000 500000\
#                                                             --tasks  SR-MMP SR-HSE SR-ATAD5 SR-ARE NR-PPAR-gamma NR-ER-LBD NR-ER NR-Aromatase NR-AhR NR-AR-LBD NR-AR SR-p53\
#                                                             --target_data tox21\
#                                                             --mol_rep smiles\
#                                                             --tokenizer_type BPE\
#                                                             --tokenizer_data smilesDB\
#                                                             --pretrained_data smilesDB\
#                                                             --epochs 40\
#                                                             --aug                                                           
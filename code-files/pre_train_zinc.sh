# !/usr/bin/env bash
# run misc. stuff
nvidia-smi
echo $CUDA_VISIBLE_DEVICES
echo $HOSTNAME
which python


# pre_training up to 40 epochs




            ################################# SMILES ZINC DATABASE ############################################

# TEST LM ON Different DATA SIZE
# LATER ON, TEST ON DIFFERENT DATA SIZE
CUDA_LAUNCH_BLOCKING=1 python pre_train_zinc.py --vocab_size 10000\
                                            --mol_rep smiles\
                                            --data_size 1000\
                                           --train_data smilesZinc\
                                           --tokenizer_data smilesZinc\
                                           --tokenizer_type BPE\
                                          --epochs 10\
                                           --no-aug

CUDA_LAUNCH_BLOCKING=1 python pre_train_zinc.py --vocab_size 10000\
                                            --mol_rep smiles\
                                            --data_size 2000\
                                             --train_data smilesZinc\
                                                --tokenizer_data smilesZinc\
                                                --tokenizer_type BPE\
                                                --epochs 10\
                                                --no-aug

CUDA_LAUNCH_BLOCKING=1 python pre_train_zinc.py --vocab_size 10000\
                                            --mol_rep smiles\
                                            --data_size 5000\
                                             --train_data smilesZinc\
                                                --tokenizer_data smilesZinc\
                                                --tokenizer_type BPE\
                                                --epochs 10\
                                                --no-aug

CUDA_LAUNCH_BLOCKING=1 python pre_train_zinc.py --vocab_size 10000\
                                            --mol_rep smiles\
                                            --data_size 10000\
                                             --train_data smilesZinc\
                                                --tokenizer_data smilesZinc\
                                                --tokenizer_type BPE\
                                                --epochs 10\
                                                --no-aug

CUDA_LAUNCH_BLOCKING=1 python pre_train_zinc.py --vocab_size 10000\
                                            --mol_rep smiles\
                                            --data_size 20000\
                                             --train_data smilesZinc\
                                                --tokenizer_data smilesZinc\
                                                --tokenizer_type BPE\
                                                --epochs 10\
                                                --no-aug

CUDA_LAUNCH_BLOCKING=1 python pre_train_zinc.py --vocab_size 10000\
                                            --mol_rep smiles\
                                            --data_size 50000\
                                             --train_data smilesZinc\
                                                --tokenizer_data smilesZinc\
                                                --tokenizer_type BPE\
                                                --epochs 10\
                                                --no-aug

CUDA_LAUNCH_BLOCKING=1 python pre_train_zinc.py --vocab_size 10000\
                                            --mol_rep smiles\
                                            --data_size 100000\
                                             --train_data smilesZinc\
                                                --tokenizer_data smilesZinc\
                                                --tokenizer_type BPE\
                                                --epochs 10\
                                                --no-aug

CUDA_LAUNCH_BLOCKING=1 python pre_train_zinc.py --vocab_size 10000\
                                            --mol_rep smiles\
                                            --data_size 200000\
                                             --train_data smilesZinc\
                                                --tokenizer_data smilesZinc\
                                                --tokenizer_type BPE\
                                                --epochs 10\
                                                --no-aug



# RUN PRE-TRAINING ON BPE TOKENIZERS
CUDA_LAUNCH_BLOCKING=1 python pre_train_zinc.py --vocab_size  100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000\
                                            --data_size 2000\
                                            --train_data smilesDB\
                                            --tokenizer_data smilesDB\
                                            --tokenizer_type BPE\
                                             --epochs 30\
                                            --no-aug

                                            


# vocab_sizes = [100, 200, 500, 1000,2000,5000,10000,20000,50000]
# PRE-TRAINING ON SMILES ATOMIC TOKENIZERS
# CUDA_LAUNCH_BLOCKING=1 python pre_train_zinc.py --vocab_size 77\
#                                             --mol_rep smiles\
#                                            --train_data smilesZinc\
#                                            --tokenizer_data smilesZinc\
#                                            --tokenizer_type Atom-wise\
#                                              --epochs 40\
#                                            --no-aug
                                         





# PRE-TRAINING ON SELFIES ATOMIC TOKENIZERS
# CUDA_LAUNCH_BLOCKING=1 python pre_train_zinc.py --vocab_size 122\
#                                            --mol_rep selfies\
#                                            --train_data selfiesZinc\
#                                            --tokenizer_data selfiesZinc\
#                                            --tokenizer_type Atom-wise\
#                                            --epochs 40\
#                                            --no-aug


# DATA-DRIVEN TOKENIZERS
# RUN PRE-TRAINING ON MORFESSOR TOKENIZERS
# CUDA_LAUNCH_BLOCKING=1 python pre_train_zinc.py --vocab_size  100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000\
#                                             --train_data smilesZinc\
#                                             --tokenizer_data smilesZinc\
#                                             --tokenizer_type Morfessor\
#                                             --epochs 40\
#                                             --no-aug

# RUN PRE-TRAINING ON WORDPIECE TOKENIZERS

# CUDA_LAUNCH_BLOCKING=1 python pre_train_zinc.py --vocab_size 100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000\
#                                             --train_data smilesZinc\
#                                             --tokenizer_data smilesZinc\
#                                             --tokenizer_type WordPiece\
#                                              --epochs 40\
#                                             --no-aug



# RUN PRE-TRAINING ON SPE TOKENIZERS
# CUDA_LAUNCH_BLOCKING=1 python pre_train_zinc.py --vocab_size  100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000\
#                                             --train_data smilesZinc\
#                                             --tokenizer_data smilesZinc\
#                                             --tokenizer_type SPE\
#                                              --epochs 40\
#                                             --no-aug

# # CHEMICAL-RULES BASED TOKENIZERS
# CUDA_LAUNCH_BLOCKING=1 python  pre_train_zinc.py  --train_data smilesZinc_mac_fragments\
#                                              --tokenizer_data smilesZinc\
#                                              --tokenizer_type MacFrag\
#                                              --epochs 40\
#                                               --no-aug

# CUDA_LAUNCH_BLOCKING=1  python pre_train_zinc.py  --train_data smilesZinc_mac_fragments\
#                                           --tokenizer_data smilesClintox\
#                                           --tokenizer_type MacFrag\
#                                           --epochs 40\
#                                           --no-aug

# CUDA_LAUNCH_BLOCKING=1 python  pre_train_zinc.py --train_data smilesZinc_mac_fragments\
#                                              --tokenizer_data smilesTox21\
#                                             --tokenizer_type MacFrag\
#                                              --epochs 40\
#                                             --no-aug

# CUDA_LAUNCH_BLOCKING=1 python pre_train_zinc.py --train_data smilesZinc_mac_fragments\
#                                                 --tokenizer_data smilesDB\
#                                                 --tokenizer_type MacFrag\
#                                                 --epochs 40\
#                                                 --no-aug

# CUDA_LAUNCH_BLOCKING=1  python pre_train_zinc.py --train_data smilesZinc_mac_fragments\
#                                             --tokenizer_data smilesPubChem500k\
#                                             --tokenizer_type MacFrag\
#                                              --epochs 40\
#                                             --no-aug


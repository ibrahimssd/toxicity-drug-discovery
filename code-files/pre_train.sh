# !/usr/bin/env bash
# run misc. stuff
nvidia-smi
echo $CUDA_VISIBLE_DEVICES
echo $HOSTNAME
which python




################################################################## SMILES ####################################################################

                            ################################# SMILES DB DATABASE ############################################

# vocab_sizes = [100, 200, 500, 1000,2000,5000,10000,20000,50000, 100000, 200000, 500000]


# RUN PRE-TRAINING ON SMILES ATOMIC TOKENIZERS
# CUDA_LAUNCH_BLOCKING=1 python pre_train.py --vocab_size 163\
#                                            --train_data smilesDB\
#                                            --tokenizer_data smilesDB\
#                                            --tokenizer_type Atom-wise





# RUN PRE-TRAINING ON SELFIES ATOMIC TOKENIZERS
# CUDA_LAUNCH_BLOCKING=1 python pre_train.py --vocab_size 204\
#                                            --train_data selfiesDB\
#                                            --tokenizer_data selfiesDB\
#                                            --tokenizer_type Atom-wise


# DATA-DRIVEN TOKENIZERS
# RUN PRE-TRAINING ON MORFESSOR TOKENIZERS
CUDA_LAUNCH_BLOCKING=1 python pre_train.py --vocab_size   100000 200000 500000\
                                            --train_data smilesDB\
                                            --tokenizer_data smilesDB\
                                            --tokenizer_type Morfessor



# RUN PRE-TRAINING ON WORDPIECE TOKENIZERS
# CUDA_LAUNCH_BLOCKING=1 python pre_train.py --vocab_size  100000 200000 500000\
#                                             --train_data smilesDB\
#                                             --tokenizer_data smilesDB\
#                                             --tokenizer_type WordPiece


# RUN PRE-TRAINING ON BPE TOKENIZERS
CUDA_LAUNCH_BLOCKING=1 python pre_train.py --vocab_size  100000 200000 500000\
                                            --train_data smilesDB\
                                            --tokenizer_data smilesDB\
                                            --tokenizer_type BPE



# # RUN PRE-TRAINING ON SPE TOKENIZERS
CUDA_LAUNCH_BLOCKING=1 python pre_train.py --vocab_size  100000 200000 500000\
                                            --train_data smilesDB\
                                            --tokenizer_data smilesDB\
                                            --tokenizer_type SPE


# CHEMICAL-RULES BASED TOKENIZERS
# MacFrag_VOCAB_SIZES = [1913: clintox , 2493 : Tox21, 31729: Zinc , 36333: DB , 114620: PubChem500k]
# CUDA_LAUNCH_BLOCKING=1 python pre_train.py  --train_data smilesDB_mac_fragments\
#                                              --tokenizer_type MacFrag\
#                                               --tokenizer_data smilesDB


# CUDA_LAUNCH_BLOCKING=1 python pre_train.py  --train_data smilesDB_mac_fragments\
#                                             --tokenizer_type MacFrag\
#                                             --tokenizer_data smilesClintox
                                             

# CUDA_LAUNCH_BLOCKING=1 python pre_train.py --train_data smilesDB_mac_fragments\
#                                             --tokenizer_type MacFrag\
#                                             --tokenizer_data smilesTox21\
                                             

# CUDA_LAUNCH_BLOCKING=1 python pre_train.py --train_data smilesDB_mac_fragments \
#                                              --tokenizer_type MacFrag\
#                                                 --tokenizer_data smilesZinc

CUDA_LAUNCH_BLOCKING=1 python pre_train.py --train_data smilesDB_mac_fragments\
                                            --tokenizer_type MacFrag\
                                            --tokenizer_data smilesPubChem500k                                             






#################################################### SELFIES ###########################################################################



# # DATA-DRIVEN TOKENIZERS
# CUDA_LAUNCH_BLOCKING=1 python pre_train.py --vocab_size 100 200 500 1000 2000 5000 10000 20000 50000\
#                                             --train_data selfiesDB\
#                                             --tokenizer_data selfiesDB\
#                                             --tokenizer_type Morfessor




# CUDA_LAUNCH_BLOCKING=1 python pre_train.py --vocab_size 100 200 500 1000 2000 5000 10000 20000 50000\
#                                             --train_data selfiesDB\
#                                             --tokenizer_data selfiesDB\
#                                             --tokenizer_type WordPiece




# CUDA_LAUNCH_BLOCKING=1 python pre_train.py --vocab_size 100 200 500 1000 2000 5000 10000 20000 50000\
#                                             --train_data selfiesDB\
#                                             --tokenizer_data selfiesDB\
#                                             --tokenizer_type BPE


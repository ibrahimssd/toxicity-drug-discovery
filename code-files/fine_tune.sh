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

# delete logs files if exists
# rm -rf ./logs/fine_tune/*
# run the code
# vocab_sizes = [100, 200, 500, 1000,2000,5000,10000,20000,50000, 100000, 200000, 500000]
#mkdir ./.cache
# FINE-TUNING ON SMILES
# ATOMIC TOKENIZERS
# SR-p53
#'NR-AR', 'NR-AR-LBD','NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
#'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']

# TESTING SNIPPET
# CUDA_LAUNCH_BLOCKING=1 python fine_tune.py --vocab_sizes  50000\
#                                             --target_data tox21\
#                                             --mol_rep smiles\
#                                             --task SR-p53\
#                                             --aux_task SR-p53\
#                                             --tokenizer_type WordPiece\
#                                             --tokenizer_data smilesDB\
#                                             --pretrained_data smilesDB\
#                                             --epochs 2\
#                                             --aug
                                             
    

#CT_TOX
# SMILES VS SELFIES
# TASK =CT_TOX
CUDA_LAUNCH_BLOCKING=1 python fine_tune.py --vocab_size 163\
                                           --target_data clintox\
                                           --mol_rep smiles\
                                           --task CT_TOX\
                                           --aux_task CT_TOX\
                                           --tokenizer_type Atom-wise\
                                           --tokenizer_data smilesDB\
                                           --pretrained_data smilesDB\
                                           --epochs 30\
                                           --aug

CUDA_LAUNCH_BLOCKING=1 python fine_tune.py --vocab_size 204\
                                           --target_data clintox\
                                           --mol_rep selfies\
                                            --task CT_TOX\
                                            --aux_task CT_TOX\
                                            --tokenizer_type Atom-wise\
                                            --tokenizer_data selfiesDB\
                                           --pretrained_data selfiesDB\
                                           --epochs 30\
                                           --aug 


CUDA_LAUNCH_BLOCKING=1 python fine_tune.py --vocab_sizes 100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000\
                                            --target_data clintox\
                                            --mol_rep smiles\
                                            --task CT_TOX\
                                            --aux_task CT_TOX\
                                            --tokenizer_type Morfessor\
                                            --tokenizer_data smilesDB\
                                            --pretrained_data smilesDB\
                                            --epochs 30\
                                            --aug

CUDA_LAUNCH_BLOCKING=1 python fine_tune.py --vocab_sizes 100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000\
                                            --target_data clintox\
                                            --mol_rep smiles\
                                            --task CT_TOX\
                                            --aux_task CT_TOX\
                                            --tokenizer_type WordPiece\
                                            --tokenizer_data smilesDB\
                                            --pretrained_data smilesDB\
                                            --epochs 30\
                                            --aug

CUDA_LAUNCH_BLOCKING=1 python fine_tune.py --vocab_sizes 100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000\
                                            --target_data clintox\
                                            --mol_rep smiles\
                                            --task CT_TOX\
                                            --aux_task CT_TOX\
                                            --tokenizer_type BPE\
                                            --tokenizer_data smilesDB\
                                            --pretrained_data smilesDB\
                                            --epochs 30\
                                            --aug

CUDA_LAUNCH_BLOCKING=1 python fine_tune.py --vocab_sizes  100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000\
                                            --target_data clintox\
                                            --mol_rep smiles\
                                            --task CT_TOX\
                                            --aux_task CT_TOX\
                                            --tokenizer_type SPE\
                                            --tokenizer_data smilesDB\
                                            --pretrained_data smilesDB\
                                            --epochs 30\
                                            --aug

# # RULES BASED TOKENIZERS
# # MacFrag_VOCAB_SIZES = [1913(1902): clintox , 2493 : Tox21, 31729: Zinc , 36333 (34146): DB  , 114620: PubChem500k]
CUDA_LAUNCH_BLOCKING=1 python fine_tune.py  --target_data clintox\
                                            --vocab_sizes 1913\
                                            --mol_rep smiles_mac_frags\
                                            --task CT_TOX\
                                            --aux_task CT_TOX\
                                            --tokenizer_type MacFrag\
                                            --tokenizer_data smilesClintox\
                                            --pretrained_data smilesDB_mac_fragments\
                                            --epochs 30\
                                            --aug

CUDA_LAUNCH_BLOCKING=1 python fine_tune.py  --target_data clintox\
                                            --vocab_sizes 2493\
                                            --mol_rep smiles_mac_frags\
                                            --task CT_TOX\
                                            --aux_task CT_TOX\
                                             --tokenizer_type MacFrag\
                                             --tokenizer_data smilesTox21\
                                            --pretrained_data smilesDB_mac_fragments\
                                            --epochs 30\
                                            --aug

CUDA_LAUNCH_BLOCKING=1 python fine_tune.py  --target_data clintox\
                                            --vocab_sizes 31729\
                                            --mol_rep smiles_mac_frags\
                                            --task CT_TOX\
                                            --aux_task CT_TOX\
                                             --tokenizer_type MacFrag\
                                             --tokenizer_data smilesZinc\
                                            --pretrained_data smilesDB_mac_fragments\
                                            --epochs 30\
                                            --aug

CUDA_LAUNCH_BLOCKING=1 python fine_tune.py  --target_data clintox\
                                            --vocab_sizes  36333\
                                            --mol_rep smiles_mac_frags\
                                            --task CT_TOX\
                                            --aux_task CT_TOX\
                                             --tokenizer_type MacFrag\
                                             --tokenizer_data smilesDB\
                                            --pretrained_data smilesDB_mac_fragments\
                                            --epochs 30\
                                            --aug

CUDA_LAUNCH_BLOCKING=1 python fine_tune.py  --target_data clintox\
                                            --vocab_sizes 114620\
                                            --mol_rep smiles_mac_frags\
                                            --task CT_TOX\
                                            --aux_task CT_TOX\
                                             --tokenizer_type MacFrag\
                                             --tokenizer_data smilesPubChem500k\
                                            --pretrained_data smilesDB_mac_fragments\
                                            --epochs 30\
                                            --aug   







# TASK = NR-Aromatase
# CUDA_LAUNCH_BLOCKING=1 python fine_tune.py --vocab_size 163\
#                                            --target_data tox21\
#                                            --mol_rep smiles\
#                                            --task SR-p53\
#                                            --aux_task SR-p53\
#                                             --tokenizer_type Atom-wise\
#                                             --tokenizer_data smilesDB\
#                                            --pretrained_data smilesDB\
#                                            --epochs 20\
#                                            --aug

# CUDA_LAUNCH_BLOCKING=1 python fine_tune.py --vocab_size 204\
#                                            --target_data tox21\
#                                            --mol_rep selfies\
#                                             --task SR-p53\
#                                             --aux_task SR-p53\
#                                             --tokenizer_type Atom-wise\
#                                             --tokenizer_data selfiesDB\
#                                            --pretrained_data selfiesDB\
#                                            --epochs 20\
#                                            --aug 






# # DATA-DRIVEN TOKENIZERS
# # 'NR-AR', 'NR-AR-LBD','NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
# # 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
# CUDA_LAUNCH_BLOCKING=1 python fine_tune.py --vocab_sizes 100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000\
#                                             --target_data tox21\
#                                             --mol_rep smiles\
#                                             --task SR-p53\
#                                             --aux_task SR-p53\
#                                             --tokenizer_type Morfessor\
#                                             --tokenizer_data smilesDB\
#                                             --pretrained_data smilesDB\
#                                             --epochs 20\
#                                             --aug
 

                                          


# CUDA_LAUNCH_BLOCKING=1 python fine_tune.py --vocab_sizes 100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000\
#                                             --target_data tox21\
#                                             --mol_rep smiles\
#                                             --task SR-p53\
#                                             --aux_task SR-p53\
#                                             --tokenizer_type WordPiece\
#                                             --tokenizer_data smilesDB\
#                                             --pretrained_data smilesDB\
#                                             --epochs 20\
#                                             --aug 
               

# CUDA_LAUNCH_BLOCKING=1 python fine_tune.py --vocab_sizes 100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000\
#                                             --target_data tox21\
#                                             --mol_rep smiles\
#                                             --task SR-p53\
#                                             --aux_task SR-p53\
#                                             --tokenizer_type BPE\
#                                             --tokenizer_data smilesDB\
#                                             --pretrained_data smilesDB\
#                                             --epochs 20\
#                                             --aug 
                               


# CUDA_LAUNCH_BLOCKING=1 python fine_tune.py --vocab_sizes  100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000\
#                                             --target_data tox21\
#                                             --mol_rep smiles\
#                                             --task SR-p53\
#                                             --aux_task SR-p53\
#                                             --tokenizer_type SPE\
#                                             --tokenizer_data smilesDB\
#                                             --pretrained_data smilesDB\
#                                             --epochs 20\
#                                             --aug 
                                            
                


# # RULES BASED TOKENIZERS
# # MacFrag_VOCAB_SIZES = [1913(1902): clintox , 2493 : Tox21, 31729: Zinc , 36333 (34146): DB  , 114620: PubChem500k]
# CUDA_LAUNCH_BLOCKING=1 python fine_tune.py  --target_data tox21\
#                                             --vocab_sizes 1913\
#                                             --mol_rep smiles_mac_frags\
#                                             --task SR-p53\
#                                             --aux_task SR-p53\
#                                             --tokenizer_type MacFrag\
#                                             --tokenizer_data smilesClintox\
#                                             --pretrained_data smilesDB_mac_fragments\
#                                             --epochs 20\
#                                             --aug                                    

# CUDA_LAUNCH_BLOCKING=1 python fine_tune.py  --target_data tox21\
#                                             --vocab_sizes 2493\
#                                             --mol_rep smiles_mac_frags\
#                                             --task SR-p53\
#                                             --aux_task SR-p53\
#                                              --tokenizer_type MacFrag\
#                                              --tokenizer_data smilesTox21\
#                                             --pretrained_data smilesDB_mac_fragments\
#                                             --epochs 20\
#                                             --aug                                       

# CUDA_LAUNCH_BLOCKING=1 python fine_tune.py  --target_data tox21\
#                                             --vocab_sizes 31729\
#                                             --mol_rep smiles_mac_frags\
#                                             --task SR-p53\
#                                             --aux_task SR-p53\
#                                              --tokenizer_type MacFrag\
#                                              --tokenizer_data smilesZinc\
#                                             --pretrained_data smilesDB_mac_fragments\
#                                             --epochs 20\
#                                             --aug                                         

# CUDA_LAUNCH_BLOCKING=1 python fine_tune.py  --target_data tox21\
#                                             --vocab_sizes  36333\
#                                             --mol_rep smiles_mac_frags\
#                                             --task SR-p53\
#                                             --aux_task SR-p53\
#                                              --tokenizer_type MacFrag\
#                                              --tokenizer_data smilesDB\
#                                             --pretrained_data smilesDB_mac_fragments\
#                                             --epochs 20\
#                                             --aug

# CUDA_LAUNCH_BLOCKING=1 python fine_tune.py  --target_data tox21\
#                                             --vocab_sizes 114620\
#                                             --mol_rep smiles_mac_frags\
#                                             --task SR-p53\
#                                             --aux_task SR-p53\
#                                              --tokenizer_type MacFrag\
#                                              --tokenizer_data smilesPubChem500k\
#                                             --pretrained_data smilesDB_mac_fragments\
#                                             --epochs 20\
#                                             --aug



  









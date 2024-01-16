#!/usr/bin/env bash
echo $CUDA_VISIBLE_DEVICES
echo $HOSTNAME
which python




# ATOM-WISE Atom-wise
CUDA_LAUNCH_BLOCKING=1 python perplexity_eval.py  --vocab_sizes  163\
                                            --mol_rep smiles\
                                            --pretrained_data smilesDB\
                                            --tokenizer_data smilesDB\
                                            --test_data smilesTox21\
                                             --target_data tox21 hips clintox\
                                            --tokenizer_type Atom-wise\


# ATOM-WISE SELFIES
CUDA_LAUNCH_BLOCKING=1 python perplexity_eval.py  --vocab_sizes  204\
                                            --mol_rep selfies\
                                            --pretrained_data selfiesDB\
                                            --tokenizer_data selfiesDB\
                                            --test_data selfiesTox21\
                                             --target_data tox21 hips clintox\
                                            --tokenizer_type Atom-wise\

# BPE
CUDA_LAUNCH_BLOCKING=1 python perplexity_eval.py  --vocab_sizes  100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000\
                                            --mol_rep smiles\
                                            --pretrained_data smilesDB\
                                            --tokenizer_data smilesDB\
                                            --test_data smilesTox21\
                                             --target_data tox21 hips clintox\
                                            --tokenizer_type BPE\



# SPE
CUDA_LAUNCH_BLOCKING=1 python perplexity_eval.py  --vocab_sizes  100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000\
                                            --mol_rep smiles\
                                            --pretrained_data smilesDB\
                                            --tokenizer_data smilesDB\
                                            --test_data smilesTox21\
                                             --target_data tox21 hips clintox\
                                            --tokenizer_type SPE\

# WORDPIECE
CUDA_LAUNCH_BLOCKING=1 python perplexity_eval.py  --vocab_sizes  100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000\
                                            --mol_rep smiles\
                                            --pretrained_data smilesDB\
                                            --tokenizer_data smilesDB\
                                            --test_data smilesTox21\
                                             --target_data tox21 hips clintox\
                                            --tokenizer_type WordPiece\




# MORFESSOR
CUDA_LAUNCH_BLOCKING=1 python perplexity_eval.py  --vocab_size  100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000\
                                            --mol_rep smiles\
                                            --pretrained_data smilesDB\
                                            --tokenizer_data smilesDB\
                                            --test_data smilesTox21\
                                             --target_data tox21 hips clintox\
                                            --tokenizer_type Morfessor\


# MacFrag
# # MacFrag_VOCAB_SIZES = [1913(1902): clintox , 2493 : Tox21, 31729: Zinc , 36333 (34146): DB  , 114620: PubChem500k]
CUDA_LAUNCH_BLOCKING=1 python perplexity_eval.py  --target_data tox21 hips clintox\
                                            --vocab_sizes 1913\
                                            --mol_rep smiles_mac_frags\
                                            --tokenizer_type MacFrag\
                                            --tokenizer_data smilesClintox\
                                            --pretrained_data smilesDB_mac_fragments\
                                            

CUDA_LAUNCH_BLOCKING=1 python perplexity_eval.py  --target_data tox21 hips clintox\
                                            --vocab_sizes 2493\
                                            --mol_rep smiles_mac_frags\
                                             --tokenizer_type MacFrag\
                                             --tokenizer_data smilesTox21\
                                            --pretrained_data smilesDB_mac_fragments\
                                            

CUDA_LAUNCH_BLOCKING=1 python perplexity_eval.py  --target_data tox21 hips clintox\
                                            --vocab_sizes 31729\
                                            --mol_rep smiles_mac_frags\
                                             --tokenizer_type MacFrag\
                                             --tokenizer_data smilesZinc\
                                            --pretrained_data smilesDB_mac_fragments\
                                            

CUDA_LAUNCH_BLOCKING=1 python perplexity_eval.py  --target_data tox21 hips clintox\
                                            --vocab_sizes  36333\
                                            --mol_rep smiles_mac_frags\
                                             --tokenizer_type MacFrag\
                                             --tokenizer_data smilesDB\
                                            --pretrained_data smilesDB_mac_fragments\
                                            

CUDA_LAUNCH_BLOCKING=1 python perplexity_eval.py  --target_data tox21 hips clintox\
                                            --vocab_sizes 114620\
                                            --mol_rep smiles_mac_frags\
                                             --tokenizer_type MacFrag\
                                             --tokenizer_data smilesPubChem500k\
                                            --pretrained_data smilesDB_mac_fragments\
                                             
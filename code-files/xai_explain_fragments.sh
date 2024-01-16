
##!/usr/bin/env bash
#'NR-AR', 'NR-AR-LBD','NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
#'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']

# [1] : NR-AR-LBD
# [2] : NR-PPAR-gamma 
# [3]: NR-AR


###################################### BEST PERFORMING TOKENIZER #######################################################################################################
CUDA_LAUNCH_BLOCKING=1 python xai_explain_fragments.py --vocab_sizes 10000 10000 5000 20000 5000 20000 100000 5000 5000 10000 5000 10000\
                                            --vocab_size 10000\
                                            --target_data tox21\
                                            --mol_rep smiles\
                                            --task NR-AR-LBD\
                                            --end_points  NR-AR-LBD\
                                            --tokenizer_type BPE\
                                            --tokenizer_data smilesDB\



#NR-ER-LBD : 20000
# CUDA_LAUNCH_BLOCKING=1 python xai_explain_fragments.py --vocab_size 20000\
#                                             --target_data tox21\
#                                             --mol_rep smiles\
#                                             --task NR-ER-LBD\
#                                             --tokenizer_type BPE\
#                                             --tokenizer_data smilesDB\










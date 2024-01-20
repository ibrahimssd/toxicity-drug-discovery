#!/bin/bash
echo "Morfessor tokenization"
#vocab_sizes = [100, 200, 500, 1000,2000,5000,10000,20000,50000, 100000, 200000, 500000]
# echo "Trining Morfessor model On Clinical Toxicity dataset"
# morfessor-train\
#  -s ./models/tokenizers/morfessors/morf_clintox_100.bin\
#  --logfile=./tokenization_algorithms/morf_logs/morf_clintox_100.log\
#  --num-morph-types=100\
#  --max-epochs=100\
#  ./datasets/pre_processed/clintox.smi \

# morfessor-train\
#  -s ./models/tokenizers/morfessors/morf_clintox_200.bin\
#  --logfile=./tokenization_algorithms/morf_logs/morf_clintox_200.log\
#  --num-morph-types=200\
#  --max-epochs=100\
#  ./datasets/pre_processed/clintox.smi \


# morfessor-train\
#  -s ./models/tokenizers/morfessors/morf_clintox_500.bin\
#  --logfile=./tokenization_algorithms/morf_logs/morf_clintox_500.log\
#  --num-morph-types=500\
#  --max-epochs=100\
#  ./datasets/pre_processed/clintox.smi\



# morfessor-train\
#  -s ./models/tokenizers/morfessors/morf_clintox_1000.bin\
#  --logfile=./tokenization_algorithms/morf_logs/morf_clintox_1000.log\
#  --num-morph-types=1000\
#  --max-epochs=100\
#  ./datasets/pre_processed/clintox.smi\


# morfessor-train\
#     -s ./models/tokenizers/morfessors/morf_clintox_2000.bin\
#     --logfile=./tokenization_algorithms/morf_logs/morf_clintox_2000.log\
#     --num-morph-types=2000\
#      --max-epochs=100\
#     ./datasets/pre_processed/clintox.smi \

# morfessor-train\
#     -s ./models/tokenizers/morfessors/morf_clintox_5000.bin\
#     --logfile=./tokenization_algorithms/morf_logs/morf_clintox_5000.log\
#     --num-morph-types=5000\
#      --max-epochs=100\
#     ./datasets/pre_processed/clintox.smi \

# morfessor-train\
#     -s ./models/tokenizers/morfessors/morf_clintox_10000.bin\
#     --logfile=./tokenization_algorithms/morf_logs/morf_clintox_10000.log\
#     --num-morph-types=10000\
#      --max-epochs=100\
#     ./datasets/pre_processed/clintox.smi \


# morfessor-train\
#     -s ./models/tokenizers/morfessors/morf_clintox_20000.bin\
#     --logfile=./tokenization_algorithms/morf_logs/morf_clintox_20000.log\
#     --num-morph-types=20000\
#      --max-epochs=100\
#     ./datasets/pre_processed/clintox.smi\

# morfessor-train\
#     -s ./models/tokenizers/morfessors/morf_clintox_50000.bin\
#     --logfile=./tokenization_algorithms/morf_logs/morf_clintox_50000.log\
#     --num-morph-types=50000\
#      --max-epochs=100\
#     ./datasets/pre_processed/clintox.smi\

# echo " Finished training Morfessor model On Clinical Toxicity dataset"




# echo "Trining Morfessor model On Tox21 dataset"
# morfessor-train\
#  -s ./models/tokenizers/morfessors/morf_tox21_100.bin\
#  --logfile=./tokenization_algorithms/morf_logs/morf_tox21_100.log\
#  --num-morph-types=100\
#  --max-epochs=100\
#  ./datasets/pre_processed/tox21.smi\


# morfessor-train\
#     -s ./models/tokenizers/morfessors/morf_tox21_200.bin\
#     --logfile=./tokenization_algorithms/morf_logs/morf_tox21_200.log\
#     --num-morph-types=200\
#     --max-epochs=100\
#     ./datasets/pre_processed/tox21.smi\

# morfessor-train\
#     -s ./models/tokenizers/morfessors/morf_tox21_500.bin\
#     --logfile=./tokenization_algorithms/morf_logs/morf_tox21_500.log\
#     --num-morph-types=500\
#     --max-epochs=100\
#     ./datasets/pre_processed/tox21.smi\

# morfessor-train\
#     -s ./models/tokenizers/morfessors/morf_tox21_1000.bin\
#     --logfile=./tokenization_algorithms/morf_logs/morf_tox21_1000.log\
#     --num-morph-types=1000\
#     --max-epochs=100\
#     ./datasets/pre_processed/tox21.smi\

# morfessor-train\
#     -s ./models/tokenizers/morfessors/morf_tox21_2000.bin\
#     --logfile=./tokenization_algorithms/morf_logs/morf_tox21_2000.log\
#     --num-morph-types=2000\
#     --max-epochs=100\
#     ./datasets/pre_processed/tox21.smi\

# morfessor-train\
#     -s ./models/tokenizers/morfessors/morf_tox21_5000.bin\
#     --logfile=./tokenization_algorithms/morf_logs/morf_tox21_5000.log\
#     --num-morph-types=5000\
#     --max-epochs=100\
#     ./datasets/pre_processed/tox21.smi\

# morfessor-train\
#     -s ./models/tokenizers/morfessors/morf_tox21_10000.bin\
#     --logfile=./tokenization_algorithms/morf_logs/morf_tox21_10000.log\
#     --num-morph-types=10000\
#     --max-epochs=100\
#     ./datasets/pre_processed/tox21.smi\

# morfessor-train\
#     -s ./models/tokenizers/morfessors/morf_tox21_20000.bin\
#     --logfile=./tokenization_algorithms/morf_logs/morf_tox21_20000.log\
#     --num-morph-types=20000\
#     --max-epochs=100\
#     ./datasets/pre_processed/tox21.smi\

# morfessor-train\
#     -s ./models/tokenizers/morfessors/morf_tox21_50000.bin\
#     --logfile=./tokenization_algorithms/morf_logs/morf_tox21_50000.log\
#     --num-morph-types=50000\
#     --max-epochs=100\
#     ./datasets/pre_processed/tox21.smi\

# echo "Finished training Morfessor model On Tox21 dataset"




# echo "Trining Morfessor model On clintox_tox21_zinc (DB) dataset"
# morfessor-train\
#  -s ./models/tokenizers/morfessors/morf_smilesDB_100.bin\
#  --logfile=./tokenization_algorithms/morf_logs/morf_smilesDB_100.log\
#  --num-morph-types=100\
#  --max-epochs=100\
#  ./datasets/smilesDB.smi\


# morfessor-train\
#     -s ./models/tokenizers/morfessors/morf_smilesDB_200.bin\
#     --logfile=./tokenization_algorithms/morf_logs/morf_smilesDB_200.log\
#     --num-morph-types=200\
#     --max-epochs=100\
#     ./datasets/smilesDB.smi\

# morfessor-train\
#     -s ./models/tokenizers/morfessors/morf_smilesDB_500.bin\
#     --logfile=./tokenization_algorithms/morf_logs/morf_smilesDB_500.log\
#     --num-morph-types=500\
#     --max-epochs=100\
#     ./datasets/smilesDB.smi\

# morfessor-train\
#     -s ./models/tokenizers/morfessors/morf_smilesDB_1000.bin\
#     --logfile=./tokenization_algorithms/morf_logs/morf_smilesDB_1000.log\
#     --num-morph-types=1000\
#     --max-epochs=100\
#     ./datasets/smilesDB.smi\

# morfessor-train\
#     -s ./models/tokenizers/morfessors/morf_smilesDB_2000.bin\
#     --logfile=./tokenization_algorithms/morf_logs/morf_smilesDB_2000.log\
#     --num-morph-types=2000\
#     --max-epochs=100\
#     ./datasets/smilesDB.smi\

# morfessor-train\
#     -s ./models/tokenizers/morfessors/morf_smilesDB_5000.bin\
#     --logfile=./tokenization_algorithms/morf_logs/morf_smilesDB_5000.log\
#     --num-morph-types=5000\
#     --max-epochs=100\
#     ./datasets/smilesDB.smi\

# morfessor-train\
#     -s ./models/tokenizers/morfessors/morf_smilesDB_10000.bin\
#     --logfile=./tokenization_algorithms/morf_logs/morf_smilesDB_10000.log\
#     --num-morph-types=10000\
#     --max-epochs=100\
#     ./datasets/smilesDB.smi\

# morfessor-train\
#     -s ./models/tokenizers/morfessors/morf_smilesDB_20000.bin\
#     --logfile=./tokenization_algorithms/morf_logs/morf_smilesDB_20000.log\
#     --num-morph-types=20000\
#     --max-epochs=100\
#     ./datasets/smilesDB.smi\

# morfessor-train\
#     -s ./models/tokenizers/morfessors/morf_smilesDB_50000.bin\
#     --logfile=./tokenization_algorithms/morf_logs/morf_smilesDB_50000.log\
#     --num-morph-types=50000\
#     --max-epochs=150\
#     ./datasets/smilesDB.smi\


morfessor-train\
    -s ./models/tokenizers/morfessors/morf_smilesDB_100000.bin\
    --logfile=./tokenization_algorithms/morf_logs/morf_smilesDB_100000.log\
    --num-morph-types=100000\
    --max-epochs=150\
    ./datasets/smilesDB.smi\


morfessor-train\
    -s ./models/tokenizers/morfessors/morf_smilesDB_200000.bin\
    --logfile=./tokenization_algorithms/morf_logs/morf_smilesDB_200000.log\
    --num-morph-types=200000\
    --max-epochs=150\
    ./datasets/smilesDB.smi\


morfessor-train\
    -s ./models/tokenizers/morfessors/morf_smilesDB_500000.bin\
    --logfile=./tokenization_algorithms/morf_logs/morf_smilesDB_500000.log\
    --num-morph-types=500000\
    --max-epochs=150\
    ./datasets/smilesDB.smi\



# echo "Finished training Morfessor model On clintox_tox21_zinc dataset"



echo "Trining Morfessor model On smiles zinc dataset"
# morfessor-train\
#  -s ./models/tokenizers/morfessors/morf_smilesZinc_100.bin\
#  --logfile=./tokenization_algorithms/morf_logs/morf_smilesZinc_100.log\
#  --num-morph-types=100\
#  --max-epochs=100\
#  ./datasets/pre_processed/smilesZinc.smi\


# morfessor-train\
#     -s ./models/tokenizers/morfessors/morf_smilesZinc_200.bin\
#     --logfile=./tokenization_algorithms/morf_logs/morf_smilesZinc_200.log\
#     --num-morph-types=200\
#     --max-epochs=100\
#     ./datasets/pre_processed/smilesZinc.smi\

# morfessor-train\
#     -s ./models/tokenizers/morfessors/morf_smilesZinc_500.bin\
#     --logfile=./tokenization_algorithms/morf_logs/morf_smilesZinc_500.log\
#     --num-morph-types=500\
#     --max-epochs=100\
#     ./datasets/pre_processed/smilesZinc.smi\

# morfessor-train\
#     -s ./models/tokenizers/morfessors/morf_smilesZinc_1000.bin\
#     --logfile=./tokenization_algorithms/morf_logs/morf_smilesZinc_1000.log\
#     --num-morph-types=1000\
#     --max-epochs=100\
#     ./datasets/pre_processed/smilesZinc.smi\

# morfessor-train\
#    -s ./models/tokenizers/morfessors/morf_smilesZinc_2000.bin\
#     --logfile=./tokenization_algorithms/morf_logs/morf_smilesZinc_2000.log\
#     --num-morph-types=2000\
#     --max-epochs=100\
#     ./datasets/pre_processed/smilesZinc.smi\

# morfessor-train\
#     -s ./models/tokenizers/morfessors/morf_smilesZinc_5000.bin\
#     --logfile=./tokenization_algorithms/morf_logs/morf_smilesZinc_5000.log\
#     --num-morph-types=5000\
#     --max-epochs=100\
#     ./datasets/pre_processed/smilesZinc.smi\

# morfessor-train\
#    -s ./models/tokenizers/morfessors/morf_smilesZinc_10000.bin\
#     --logfile=./tokenization_algorithms/morf_logs/morf_smilesZinc_10000.log\
#     --num-morph-types=10000\
#     --max-epochs=100\
#     ./datasets/pre_processed/smilesZinc.smi\

# morfessor-train\
#     -s ./models/tokenizers/morfessors/morf_smilesZinc_20000.bin\
#     --logfile=./tokenization_algorithms/morf_logs/morf_smilesZinc_20000.log\
#     --num-morph-types=20000\
#     --max-epochs=150\
#     ./datasets/pre_processed/smilesZinc.smi\

# morfessor-train\
#     -s ./models/tokenizers/morfessors/morf_smilesZinc_50000.bin\
#     --logfile=./tokenization_algorithms/morf_logs/morf_smilesZinc_50000.log\
#     --num-morph-types=50000\
#     --max-epochs=150\
#     ./datasets/pre_processed/smilesZinc.smi\


morfessor-train\
    -s ./models/tokenizers/morfessors/morf_smilesZinc_100000.bin\
    --logfile=./tokenization_algorithms/morf_logs/morf_smilesZinc_100000.log\
    --num-morph-types=100000\
    --max-epochs=150\
    ./datasets/pre_processed/smilesZinc.smi\


morfessor-train\
    -s ./models/tokenizers/morfessors/morf_smilesZinc_200000.bin\
    --logfile=./tokenization_algorithms/morf_logs/morf_smilesZinc_200000.log\
    --num-morph-types=200000\
    --max-epochs=150\
    ./datasets/pre_processed/smilesZinc.smi\


morfessor-train\
    -s ./models/tokenizers/morfessors/morf_smilesZinc_500000.bin\
    --logfile=./tokenization_algorithms/morf_logs/morf_smilesZinc_500000.log\
    --num-morph-types=500000\
    --max-epochs=150\
    ./datasets/pre_processed/smilesZinc.smi\

    
    
echo "Finished training Morfessor model On zinc dataset"




#!/bin/bash
echo "Morfessor tokenization"

echo "Trining Morfessor model On Clinical Toxicity dataset"
morfessor-train\
 -s ../models/tokenizers/morf_clintox_100.bin\
 --logfile=morf_logs/morf_clintox_100.log\
 --num-morph-types=100\
 --max-epochs=50\
 ../datasets/pre_processed/clintox.smi \

echo "Trining Morfessor model On Tox21 dataset"
morfessor-train\
 -s ../models/tokenizers/morf_tox21_100.bin\
 --logfile=morf_logs/morf_tox21_100.log\
 --num-morph-types=100\
 --max-epochs=50\
 ../datasets/tox21.smi\



echo "Trining Morfessor model On Clinical Toxicity dataset"
morfessor-train\
 -s ../models/tokenizers/morf_clintox_300.bin\
 --logfile=morf_logs/morf_clintox_300.log\
 --num-morph-types=300\
 --max-epochs=50\
 ../datasets/pre_processed/clintox.smi \

echo "Trining Morfessor model On Tox21 dataset"
morfessor-train\
 -s ../models/tokenizers/morf_tox21_300.bin\
 --logfile=morf_logs/morf_tox21_300.log\
 --num-morph-types=300\
 --max-epochs=50\
 ../datasets/tox21.smi\



echo "Trining Morfessor model On Clinical Toxicity dataset"
morfessor-train\
 -s ../models/tokenizers/morf_clintox_500.bin\
 --logfile=morf_logs/morf_clintox_500.log\
 --num-morph-types=500\
 --max-epochs=60\
 ../datasets/pre_processed/clintox.smi \

echo "Trining Morfessor model On Tox21 dataset"
morfessor-train\
 -s ../models/tokenizers/morf_tox21_500.bin\
 --logfile=morf_logs/morf_tox21_500.log\
 --num-morph-types=500\
 --max-epochs=60\
 ../datasets/tox21.smi\


echo "Trining Morfessor model On Clinical Toxicity dataset"
morfessor-train\
 -s ../models/tokenizers/morf_clintox_800.bin\
 --logfile=morf_logs/morf_clintox_800.log\
 --num-morph-types=800\
 --max-epochs=100\
 ../datasets/pre_processed/clintox.smi \

echo "Trining Morfessor model On Tox21 dataset"
morfessor-train\
 -s ../models/tokenizers/morf_tox21_800.bin\
 --logfile=morf_logs/morf_tox21_800.log\
 --num-morph-types=800\
 --max-epochs=100\
 ../datasets/tox21.smi\


echo "Trining Morfessor model On Clinical Toxicity dataset"
morfessor-train\
 -s ../models/tokenizers/morf_clintox_1500.bin\
 --logfile=morf_logs/morf_clintox_1500.log\
 --num-morph-types=1500\
 --max-epochs=100\
 ../datasets/pre_processed/clintox.smi \

echo "Trining Morfessor model On Tox21 dataset"
morfessor-train\
 -s ../models/tokenizers/morf_tox21_1500.bin\
 --logfile=morf_logs/morf_tox21_1500.log\
 --num-morph-types=1500\
 --max-epochs=100\
 ../datasets/tox21.smi\

echo "Trining Morfessor model On Clinical Toxicity dataset"
morfessor-train\
 -s ../models/tokenizers/morf_clintox_2000.bin\
 --logfile=morf_logs/morf_clintox_2000.log\
 --num-morph-types=2000\
 --max-epochs=100\
 ../datasets/pre_processed/clintox.smi \

echo "Trining Morfessor model On Tox21 dataset"
morfessor-train\
 -s ../models/tokenizers/morf_tox21_2000.bin\
 --logfile=morf_logs/morf_tox21_2000.log\
 --num-morph-types=2000\
 --max-epochs=100\
 ../datasets/tox21.smi\


# echo "Trining Morfessor model On Zinc dataset"
# morfessor-train\
#  -s ../models/tokenizers/morf_zinc_1000.bin\
#  --logfile=morf_logs/morf_tox21_1000.log\
#  --num-morph-types=1000\
#  ../datasets/zinc.smi\

# echo "Trining Morfessor model On clintox_tox21_zinc dataset"
# morfessor-train\
#  -s ../models/tokenizers/morf_smilesDB_1000.bin\
#  --logfile=morf_logs/morf_smilesDB_1000.log\
#  --num-morph-types=1000\
#  ../datasets/cleaned_smilesDB.smi\
#       # --max-epochs=200\

   


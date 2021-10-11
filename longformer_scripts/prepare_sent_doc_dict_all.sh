#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=1G # Memory
#SBATCH --ignore-pbs
#SBATCH --output=preprocess-detok.log

TRAIN_SENT_DOC_ALIGN='/home/usuaris/scratch/christine.raouf.saad/news_v11/NCv11_en-de/train_sent_doc_align.h5'
VALID_SENT_DOC_ALIGN='/home/usuaris/scratch/christine.raouf.saad/news_v11/NCv11_en-de/newstest2015_sen_doc_alignment.h5'
TEST_SENT_DOC_ALIGN='/home/usuaris/scratch/christine.raouf.saad/news_v11/NCv11_en-de/newstest2016_sen_doc_alignment.h5'
ALL_SENT_DOC_ALIGN='/home/usuaris/scratch/christine.raouf.saad/news_v11/NCv11_en-de/all_sen_doc_alignment.h5'

python3 prepare_sen_doc_dict.py  --dict_training_path $TRAIN_SENT_DOC_ALIGN  --dict_valid_path $VALID_SENT_DOC_ALIGN \
--dict_test_path  $TEST_SENT_DOC_ALIGN --dict_sent_doc_all_path $ALL_SENT_DOC_ALIGN
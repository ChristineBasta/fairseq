#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs
#SBATCH --output=prepare_data

TRAIN_H5='/home/christine/news-commentary/train__mean_classify.h5'
VALID_H5='/home/christine/news-commentary/valid_mean_classify.h5'
TEST_H5='/home/christine/news-commentary/test_mean_classify.h5'
SAVE_H5='/home/christine/news-commentary/all_lf_classify.h5'

#training data
python3 prepare_h5_lf_reps.py --dict_training_path $TRAIN_H5  --dict_valid_path $VALID_H5 \
 --dict_test_path $TEST_H5  --file_to_save $SAVE_H5


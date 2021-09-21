#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs
#SBATCH --output=prepare_data

WINOMT_FILE=''
LF_DICT_FILE=''
SENT_DOC_FILE=''
KIND_REPS='1'

#training data
python3 longformer_winomt_similar.py --winomt_file $WINOMT_FILE  --longformer_dict_file $LF_DICT_FILE \
 --sen_doc_alignment $SENT_DOC_FILE  --kind_reps $KIND_REPS

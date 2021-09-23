#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs
#SBATCH --output=winomt_lf_all.log

WINOMT_FILE='/home/usuaris/scratch/christine.raouf.saad/winomt/original_sentences/en_all_sen.txt'
LF_DICT_FILE='/home/usuaris/scratch/christine.raouf.saad/winomt/original_sentences/lf_representations_wino_all_1.h5'
SENT_DOC_FILE='/home/usuaris/scratch/christine.raouf.saad/winomt/original_sentences/sen_doc_alignment_all_dic_1.h5'
KIND_REPS='1'

#training data
python3 longformer_winomt_similar.py --winomt_file $WINOMT_FILE  --longformer_dict_file $LF_DICT_FILE \
 --sen_doc_alignment $SENT_DOC_FILE  --kind_reps $KIND_REPS

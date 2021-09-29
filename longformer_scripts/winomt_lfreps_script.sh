#!/bin/bash


#SBATCH -p veu-fast # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=40G # Memory
#SBATCH --ignore-pbs
#SBATCH --output=winomt_lf_all.log


WINOMT_FILE='/home/usuaris/scratch/christine.raouf.saad/winomt/original_sentences/en_all_sen.txt'
LF_DICT_FILE='/home/usuaris/scratch/christine.raouf.saad/winomt/original_sentences/lf_representations_h5.h5'
LF_FINAL_FILE='/home/usuaris/scratch/christine.raouf.saad/winomt/original_sentences/lf_representations_wino_all_1.h5'
SENT_DOC_FILE='/home/usuaris/scratch/christine.raouf.saad/winomt/original_sentences/sen_doc_alignment_all_dic_1.h5'
KIND_REPS='1'

#training data
python3 longformer_winomt_similar.py --winomt_file $WINOMT_FILE  --longformer_h5_file $LF_H5_FILE \
 --sen_doc_alignment $SENT_DOC_FILE  --kind_reps $KIND_REPS  --longformer_final_file $LF_FINAL_FILE

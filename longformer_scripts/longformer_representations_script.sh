#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs
#SBATCH --output=prepare_data

EXT='en'
FOLDER_PATH='' #English path
SAVE_REP_FILE='' #Reps_path
KIND_REP=1
DOC_TEXT_V_PATH=''
DOC_TEXT_T_PATH=''
SAVE_REP_V_FILE=''
SAVE_REP_T_FILE=''
SAVE_REP_ALL=''

python doc_representations.py --extension $EXT --folder_represent $FOLDER_PATH \
  --save_reps $SAVE_REP_FILE --kind_reps $KIND_REP --doc_dic_valid $DOC_TEXT_V_PATH \
  --doc_dic_test $DOC_TEXT_T_PATH  --save_reps_valid $SAVE_REP_V_FILE \
  --save_reps_tesT $SAVE_REP_T_FILE --save_reps_all $SAVE_REP_ALL
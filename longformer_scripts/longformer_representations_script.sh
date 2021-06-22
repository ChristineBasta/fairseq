#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=100G # Memory
#SBATCH --ignore-pbs
#SBATCH --output=prepare_data

EXT='en'
FOLDER_PATH='/home/usuaris/veu/christine.raouf.saad/news_micro_2/English' #English path
#SAVE_REP_FILE='/home/usuaris/veu/christine.raouf.saad/news_micro_2/train_reps.h5' #Reps_path
KIND_REP=1
WHICH_SET=2
DOC_TEXT_V_PATH='/home/usuaris/veu/christine.raouf.saad/news_micro_2/newstest2017_doc_text.h5'
DOC_TEXT_T_PATH='/home/usuaris/veu/christine.raouf.saad/news_micro_2/newstest2015_doc_text.h5'
#SAVE_REP_V_FILE='/home/usuaris/veu/christine.raouf.saad/news_micro_2/valid_reps.h5'
#SAVE_REP_T_FILE='/home/usuaris/veu/christine.raouf.saad/news_micro_2/test_reps.h5'
#SAVE_REP_ALL='/home/usuaris/veu/christine.raouf.saad/news_micro_2/all_lf_reps.h5'

python doc_representations.py --extension $EXT --folder_represent $FOLDER_PATH \
  --kind_reps $KIND_REP --doc_dic_valid $DOC_TEXT_V_PATH \
  --doc_dic_test $DOC_TEXT_T_PATH --which_file_reps $WHICH_SET

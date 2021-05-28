#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=5G # Memory
#SBATCH --ignore-pbs
#SBATCH --output=train-europarl-multi-ru.log


WORKING_DIR="/home/christine/news_micro_v/model"  #data on which we will work on
CP_DIR="checkpoint/longformer"
PYTHON="/home/christine/anaconda3/envs/fairseq_new_env/bin/python" #python library
FAIRSEQ_DIR="/home/christine/PycharmProjects/fairseq" #fairseq directory
SRC='src'
TGT='tgt'
#MAX_TOKENS=2000
MAX_TOKENS=300
#SAVE_UPDATES=30000
SAVE_UPDATES=100
mkdir -p $CP_DIR

stdbuf -i0 -e0 -o0 fairseq-train $WORKING_DIR \
  --arch  transformer_big  --optimizer adam --adam-betas '(0.9, 0.98)' \
  --clip-norm 0.0 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 \
  --warmup-updates 4000 --lr 0.001 --min-lr 1e-09 --dropout 0.1 \
  --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --max-tokens $MAX_TOKENS --update-freq 16 --save-dir $CP_DIR --save-interval-updates $SAVE_UPDATES \
  --source-lang $SRC --target-lang $TGT


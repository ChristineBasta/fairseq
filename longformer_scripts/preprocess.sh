#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs
#SBATCH --output=preprocess-detok.log


WORKING_DIR="/home/usuaris/veu/cescola/wmt20/detok"
SRC="ta"
TGT="dk"

TRN_PREF="train"
VAL_PREF="valid"
TES_PREF="test"
PYTHON="/home/usuaris/veu/cescola/virtualenv-16.0.0/torch/bin/python"
FAIRSEQ_DIR="/home/usuaris/veu/cescola/fairseq"

DEST_DIR="data-bin/eu"

mkdir $DEST_DIR

N_OP=16000


echo "apply bpe to " $SRC

subword-nmt learn-bpe -s $N_OP < ${WORKING_DIR}/${TRN_PREF}.${SRC} > ${DEST_DIR}/${TRN_PREF}.codes.${SRC}
subword-nmt apply-bpe -c  ${DEST_DIR}/${TRN_PREF}.codes.${SRC} < ${WORKING_DIR}/${TRN_PREF}.${SRC} >  ${DEST_DIR}/${TRN_PREF}.bpe.${SRC}
subword-nmt apply-bpe -c  ${DEST_DIR}/${TRN_PREF}.codes.${SRC} < ${WORKING_DIR}/${VAL_PREF}.${SRC} >  ${DEST_DIR}/${VAL_PREF}.bpe.${SRC}
subword-nmt apply-bpe -c  ${DEST_DIR}/${TRN_PREF}.codes.${SRC} < ${WORKING_DIR}/${TES_PREF}.${SRC} >  ${DEST_DIR}/${TES_PREF}.bpe.${SRC}

echo "apply bpe to " $TGT
subword-nmt learn-bpe -s $N_OP < ${WORKING_DIR}/${TRN_PREF}.${TGT} > ${DEST_DIR}/${TRN_PREF}.codes.${TGT}
subword-nmt apply-bpe -c  ${DEST_DIR}/${TRN_PREF}.codes.${SRC} < ${WORKING_DIR}/${TRN_PREF}.${TGT} >  ${DEST_DIR}/${TRN_PREF}.bpe.${TGT}
subword-nmt apply-bpe -c  ${DEST_DIR}/${TRN_PREF}.codes.${SRC} < ${WORKING_DIR}/${VAL_PREF}.${TGT} >  ${DEST_DIR}/${VAL_PREF}.bpe.${TGT}
subword-nmt apply-bpe -c  ${DEST_DIR}/${TRN_PREF}.codes.${SRC} < ${WORKING_DIR}/${TES_PREF}.${TGT} >  ${DEST_DIR}/${TES_PREF}.bpe.${TGT}

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
    --trainpref $DEST_DIR/${TRN_PREF}.bpe --validpref $DEST_DIR/${VAL_PREF}.bpe --testpref $DEST_DIR/${TES_PREF}.bpe \
    --destdir $DEST_DIR  --nwordstgt $N_OP --nwordssrc $N_OP



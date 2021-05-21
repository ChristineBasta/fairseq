#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs
#SBATCH --output=preprocess-detok.log
echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

WORKING_DIR="/home/christine/news_micro_v/data"   #data
SRC="en"
TGT="de"

TRN_PREF="train.tok.tc"
VAL_PREF="valid.tok.tc"
TES_PREF="test.tok.tc"
PYTHON="/home/christine/anaconda3/envs/fairseq_new_env/bin/python" #python library
FAIRSEQ_DIR="/home/christine/PycharmProjects/fairseq" #fairseq directory

DEST_DIR="/home/christine/news_micro_v/model"  #data final library

mkdir $DEST_DIR

N_OP=1000

BPEROOT='../subword-nmt/subword_nmt'
echo "apply bpe to " $SRC

echo "learn_bpe.py on ${TRAIN}..."


$BPEROOT/learn_bpe.py -s $N_OP < ${WORKING_DIR}/${TRN_PREF}.${SRC} > ${DEST_DIR}/${TRN_PREF}.codes.${SRC}
$BPEROOT/apply_bpe.py -c  ${DEST_DIR}/${TRN_PREF}.codes.${SRC} < ${WORKING_DIR}/${TRN_PREF}.${SRC} >  ${DEST_DIR}/${TRN_PREF}.bpe.${SRC}
$BPEROOT/apply_bpe.py -c  ${DEST_DIR}/${TRN_PREF}.codes.${SRC} < ${WORKING_DIR}/${VAL_PREF}.${SRC} >  ${DEST_DIR}/${VAL_PREF}.bpe.${SRC}
$BPEROOT/apply_bpe.py -c  ${DEST_DIR}/${TRN_PREF}.codes.${SRC} < ${WORKING_DIR}/${TES_PREF}.${SRC} >  ${DEST_DIR}/${TES_PREF}.bpe.${SRC}

echo "apply bpe to " $TGT
$BPEROOT/learn_bpe.py -s $N_OP < ${WORKING_DIR}/${TRN_PREF}.${TGT} > ${DEST_DIR}/${TRN_PREF}.codes.${TGT}
$BPEROOT/apply_bpe.py -c  ${DEST_DIR}/${TRN_PREF}.codes.${SRC} < ${WORKING_DIR}/${TRN_PREF}.${TGT} >  ${DEST_DIR}/${TRN_PREF}.bpe.${TGT}
$BPEROOT/apply_bpe.py -c  ${DEST_DIR}/${TRN_PREF}.codes.${SRC} < ${WORKING_DIR}/${VAL_PREF}.${TGT} >  ${DEST_DIR}/${VAL_PREF}.bpe.${TGT}
$BPEROOT/apply_bpe.py -c  ${DEST_DIR}/${TRN_PREF}.codes.${SRC} < ${WORKING_DIR}/${TES_PREF}.${TGT} >  ${DEST_DIR}/${TES_PREF}.bpe.${TGT}

  # binarize data
fairseq-preprocess \
    --source-lang ${SRC} --target-lang ${TGT} \
    --trainpref ${DEST_DIR}/${TRN_PREF}.bpe \
    --validpref ${DEST_DIR}/${VAL_PREF}.bpe \
    --testpref ${DEST_DIR}/${TES_PREF}.bpe \
    --destdir $DEST_DIR \
    --joined-dictionary \
    --workers 4

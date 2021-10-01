#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs
#SBATCH --output=preprocess-detok.log
echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git


SRC="en"
TGT="es"

GENERATE_PREF="generate.tok.tc"
MODEL_DIR="/home/usuaris/scratch/christine.raouf.saad/news_v11/NCv11_en-es/lftransfomer_model_baseline"  #MODEL
DEST_DIR="/home/usuaris/scratch/christine.raouf.saad/google_dataset/data"
TRN_PREF="train.tok.tc"


N_OP=1000

BPEROOT='subword-nmt/subword_nmt'
#we do not learn...we apply, we learn only
echo "apply bpe to " $SRC

echo "learn_bpe.py on ${TRAIN}..."


#$BPEROOT/learn_bpe.py -s $N_OP < ${WORKING_DIR}/${TRN_PREF}.${SRC} > ${DEST_DIR}/${TRN_PREF}.codes.${SRC}
$BPEROOT/apply_bpe.py -c  ${MODEL_DIR}/${TRN_PREF}.codes.${SRC} < ${DEST_DIR}/${GENERATE_PREF}.${SRC} >  ${DEST_DIR}/${GENERATE_PREF}.bpe.${SRC}


echo "apply bpe to " $TGT
#$BPEROOT/learn_bpe.py -s $N_OP < ${WORKING_DIR}/${TRN_PREF}.${TGT} > ${DEST_DIR}/${TRN_PREF}.codes.${TGT}
$BPEROOT/apply_bpe.py -c  ${MODEL_DIR}/${TRN_PREF}.codes.${SRC} < ${DEST_DIR}/${GENERATE_PREF}.${TGT} >  ${DEST_DIR}/${GENERATE_PREF}.bpe.${TGT}


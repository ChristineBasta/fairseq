#!/bin/bash
#SBATCH -p veu-fast # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=10G # Memory 
#SBATCH --ignore-pbs
#SBATCH --output=interactive_news_es.log
SRC="en"
TGT="es"

SAVE="checkpoint/trans-IWSLT-arch-es-1"

DEST_DIR="/home/usuaris/scratch/christine.raouf.saad/news_v11/NCv11_en-es/lftransfomer_model"


CP="checkpoint_best.pt"


input="/home/usuaris/scratch/christine.raouf.saad/winomt/winomt_doc_token/winomt_bpe_news_es/en-inter.doc.bpe.en"

output="/home/usuaris/scratch/christine.raouf.saad/winomt/winomt_doc_token/translations/lf_news_es_1/translations_all_winomt.es"
#Apply multilingual source codes to the data
 #python library
LONGFORMER_DICT="/home/usuaris/scratch/christine.raouf.saad/winomt/original_sentences/lf_representations_wino_all.h5"
SENT_DOC_ALIGN='/home/usuaris/scratch/christine.raouf.saad/winomt/original_sentences/sen_doc_alignment_all_dic.h5' 


CUDA_VISIBLE_DEVICES="" fairseq-interactive $DEST_DIR \
    --path $SAVE/$CP \
   --beam 5 --batch-size 1 --task translation_lf --remove-bpe --lf-path $LONGFORMER_DICT --sen-doc $SENT_DOC_ALIGN < $input > $output

python clean-output.py < $output > $output.cl


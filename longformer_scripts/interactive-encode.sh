#!/bin/bash


#SBATCH -p veu-fast # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=interactive-encoder.log 

CP="checkpoint_best.pt"
LONGFORMER_DICT="/home/usuaris/scratch/christine.raouf.saad/winomt/original_sentences/lf_representations_wino_all_1.h5"
SENT_DOC_ALIGN='/home/usuaris/scratch/christine.raouf.saad/winomt/original_sentences/sen_doc_alignment_all_1.h5'

encode() {
    INPUT_DATA=$1
    OUTPUT=$2
    SRC=$3
    TGT=$4
    TASK=$5
 
    cuda_visible_devices="" stdbuf -i0 -e0 -o0 python interactive_encode.py $DEST_DIR --path $CP_DIR/$CP \
         --batch-size 1 --source-lang ${SRC} --target-lang ${TGT} --task translation \
         --remove-bpe < $INPUT_DATA 2> $OUTPUT
}

encode_lf() {
    INPUT_DATA=$1
    OUTPUT=$2
    SRC=$3
    TGT=$4
    TASK=$5
 
    cuda_visible_devices="" stdbuf -i0 -e0 -o0 python interactive_encode.py $DEST_DIR --path $CP_DIR/$CP \
         --batch-size 1 --source-lang ${SRC} --target-lang ${TGT} --task translation_lf \
         --lf-path $LONGFORMER_DICT --sen-doc $SENT_DOC_ALIGN \
         --remove-bpe < $INPUT_DATA 2> $OUTPUT
}

mkdir -p /veu/cescola/christine/encodings/ 

#News Baseline 
CP_DIR="/home/usuaris/veu/christine.raouf.saad/fairseq/checkpoint/trans-IWSLT-arch_baseline-es/"
DEST_DIR="/home/usuaris/scratch/christine.raouf.saad/news_v11/NCv11_en-es/lftransfomer_model_baseline"
DATA="/home/usuaris/scratch/christine.raouf.saad/winomt/winomt_doc_token/winomt_bpe_news_baseline_es/en-inter.doc.bpe.en"

encode $DATA encodings/encodings-news-baseline.json en es

#News Skip Connection
CP_DIR="/home/usuaris/veu/christine.raouf.saad/fairseq/checkpoint/trans-news-skip-connection-es-after-both-2/"
DEST_DIR="/home/usuaris/scratch/christine.raouf.saad/news_v11/NCv11_en-es/lftransfomer_model_baseline"
DATA="/home/usuaris/scratch/christine.raouf.saad/winomt/winomt_doc_token/winomt_bpe_news_baseline_es/en-inter.doc.bpe.en"

encode $DATA encodings/encodings-news-sc.json en es

#News Longformer 
CP_DIR="/home/usuaris/veu/christine.raouf.saad/fairseq/checkpoint/trans-IWSLT-arch-es-1/"
DEST_DIR="/home/usuaris/scratch/christine.raouf.saad/news_v11/NCv11_en-es/lftransfomer_model"
DATA="/home/usuaris/scratch/christine.raouf.saad/winomt/winomt_doc_token/winomt_bpe_news_es/en-inter.doc.bpe.en"

encode_lf $DATA encodings/encodings-news-lf.json en es 


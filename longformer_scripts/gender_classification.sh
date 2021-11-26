#!/bin/bash


#SBATCH -p veu-fast # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=5G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=gender_classification.log 


seed=$(($RANDOM));

echo $seed

echo ''
echo 'News Baseline'
echo '************************************************'
echo ''
python gender_classification.py -e encodings/encodings-news-baseline.json \
    -v /home/usuaris/scratch/christine.raouf.saad/news_v11/NCv11_en-es/lftransfomer_model_baseline/dict.en.txt \
    -l /scratch/carlos/mt_gender/raw_data/en.txt \
    -o 0 \
    -s $seed


echo ''
echo 'News Skip Connection'
echo '************************************************'
echo ''
python gender_classification.py -e encodings/encodings-news-sc.json \
    -v /home/usuaris/scratch/christine.raouf.saad/news_v11/NCv11_en-es/lftransfomer_model_baseline/dict.en.txt \
    -l /scratch/carlos/mt_gender/raw_data/en.txt \
    -o 0 \
    -s $seed




echo ''
echo 'News Longformer'
echo '************************************************'
echo ''
python gender_classification.py -e encodings/encodings-news-lf.json \
    -v /home/usuaris/scratch/christine.raouf.saad/news_v11/NCv11_en-es/lftransfomer_model/dict.en.txt \
    -l /scratch/carlos/mt_gender/raw_data/en.txt \
    -o 1 \
    -s $seed



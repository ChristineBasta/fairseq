#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs
#SBATCH --output=prepare_data

eng_dir= '/home/christine/Europarl/English'
lang_dir= '/home/christine/Europarl/German'
extension= 'en'
extension_lang='de'
file_sent_doc='/home/christine/Europarl/file_sent_train'
eng_file_all_name='/home/christine/Europarl/all.en'
lang_pair_file_all_name='/home/christine/Europarl/all.de'
stats_file='/home/christine/Europarl/stats.en'
max_no_lines=250

python3 data_prepare_wmt.py --eng_dir $eng_dir  --lang_dir $lang_dir --extension $extension \
--file_sent_doc $file_sent_doc --eng_file_all_name $eng_file_all_name \
--lang_pair_file_all_name $lang_pair_file_all_name   \
--extension_lang $extension_lang --stats_file $stats_file --max_no_lines $max_no_lines

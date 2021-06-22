#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs
#SBATCH --output=prepare_data

ENG_DIR='/home/christine/news-commentary/aligned/German-English/English'
LANG_DIR='/home/christine/news-commentary/aligned/German-English/German'
EX='en'
EX_LANG='de'
ENG_DIR_NEW='/home/christine/news-commentary/aligned/German-English/English_CLEAN'
LANG_DIR_NEW='/home/christine/news-commentary/aligned/German-English/German_CLEAN'

#training data
python3 preprocess_clean_directory.py --eng_dir $ENG_DIR  --lang_dir $LANG_DIR --extension $EX\
  --extension_lang $EX_LANG --eng_new_dir $STATS_FILE --lang_new_dir $MAX_LINES

#steps to remove <P> lines, <Author> lines and empty lines
cd 'home/christine/news-commentary/aligned/German-English'
find English_CLEAN -type f -exec sed -i '/<P>/d' {} \;
find German_CLEAN -type f -exec sed -i '/<P>/d' {} \;
find English_CLEAN -type f -exec sed -i '/<AUTHOR/d' {} \;
find German_CLEAN -type f -exec sed -i '/<AUTHOR/d' {} \;
find English_CLEAN -type f -exec sed -i '/^$/d' {} \;
find German_CLEAN -type f -exec sed -i '/^$/d' {} \;




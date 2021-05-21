#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs
#SBATCH --output=prepare_data

ENG_DIR='/home/christine/news_micro_v/English'
LANG_DIR='/home/christine/news_micro_v/German'
EX='en'
EX_LANG='de'
FILE_SENT='/home/christine/news_micro_v/file_sent_train'
ENG_ALL='/home/christine/news_micro_v/all.en'
LANG_ALL='/home/christine/news_micro_v/all.de'
STATS_FILE='/home/christine/news_micro_v/stats.en'
MAX_LINES=250
#numbering=true


#training data
python3 data_prepare_wmt.py --eng_dir $ENG_DIR  --lang_dir $LANG_DIR --extension $EX\
  --file_sent_doc $FILE_SENT --eng_file_all_name $ENG_ALL \
  --lang_pair_file_all_name $LANG_ALL   \
  --extension_lang $EX_LANG --stats_file $STATS_FILE --max_no_lines $MAX_LINES \
#  --numbering $numbering


#valid data
SRC_FILE='/home/christine/news_micro_v/test/newstest2017-ende-src.en.sgm'
REF_FILE='/home/christine/news_micro_v/test/newstest2017-ende-ref.de.sgm'
SRC_WRITE_FILE='/home/christine/news_micro_v/newstest2017-ende-src.en'
REF_WRITE_FILE='/home/christine/news_micro_v/newstest2017-ende-ref.de'
DOC_TEXT_FILE='/home/christine/news_micro_v/newstest2017_doc_text'
sen_doc_align_file='/home/christine/news_micro_v/newstest2017_sen_doc_alignment'

python3 test_dev_prepare.py --file_read_src $SRC_FILE --file_read_ref $REF_FILE \
--file_write_src $SRC_WRITE_FILE --file_write_ref $REF_WRITE_FILE \
--doc_text_save  $DOC_TEXT_FILE --sen_doc_align $sen_doc_align_file

#test data
SRC_FILE='/home/christine/news_micro_v/dev/newstest2015-ende-src.en.sgm'
REF_FILE='/home/christine/news_micro_v/dev/newstest2015-ende-ref.de.sgm'
SRC_WRITE_FILE='/home/christine/news_micro_v/newstest2015-ende-src.en'
REF_WRITE_FILE='/home/christine/news_micro_v/newstest2015-ende-ref.de'
DOC_TEXT_FILE='/home/christine/news_micro_v/newstest2015_doc_text'
sen_doc_align_file='/home/christine/news_micro_v/newstest2015_sen_doc_alignment'

python3 test_dev_prepare.py --file_read_src $SRC_FILE --file_read_ref $REF_FILE \
--file_write_src $SRC_WRITE_FILE --file_write_ref $REF_WRITE_FILE \
--doc_text_save  $DOC_TEXT_FILE --sen_doc_align $sen_doc_align_file



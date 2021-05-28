#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs
#SBATCH --output=prepare_data

ENG_DIR='/home/christine/news_micro_2/English'
LANG_DIR='/home/christine/news_micro_2/German'
EX='en'
EX_LANG='de'
#dictionary file
TRAIN_SENT_DOC_ALIGN='/home/christine/news_micro_2/train_sent_doc_align.h5'
ENG_ALL='/home/christine/news_micro_2/all.en'
LANG_ALL='/home/christine/news_micro_2/all.de'
STATS_FILE='/home/christine/news_micro_2/stats.en'
MAX_LINES=250
numbering=0


#training data
python3 data_prepare_wmt.py --eng_dir $ENG_DIR  --lang_dir $LANG_DIR --extension $EX\
  --file_sent_doc $TRAIN_SENT_DOC_ALIGN --eng_file_all_name $ENG_ALL \
  --lang_pair_file_all_name $LANG_ALL   \
  --extension_lang $EX_LANG --stats_file $STATS_FILE --max_no_lines $MAX_LINES \
  --numbering $numbering


#valid data
SRC_FILE='/home/christine/news_micro_v/test/newstest2017-ende-src.en.sgm'
REF_FILE='/home/christine/news_micro_v/test/newstest2017-ende-ref.de.sgm'
SRC_WRITE_FILE='/home/christine/news_micro_2/newstest2017-ende-src.en'
REF_WRITE_FILE='/home/christine/news_micro_2/newstest2017-ende-ref.de'
DOC_TEXT_FILE='/home/christine/news_micro_2/newstest2017_doc_text.h5'
VALID_SENT_DOC_ALIGN='/home/christine/news_micro_2/newstest2017_sen_doc_alignment.h5'

python3 test_dev_prepare.py --file_read_src $SRC_FILE --file_read_ref $REF_FILE \
--file_write_src $SRC_WRITE_FILE --file_write_ref $REF_WRITE_FILE \
--doc_text_save  $DOC_TEXT_FILE --sen_doc_align $VALID_SENT_DOC_ALIGN

#test data
SRC_FILE='/home/christine/news_micro_v/dev/newstest2015-ende-src.en.sgm'
REF_FILE='/home/christine/news_micro_v/dev/newstest2015-ende-ref.de.sgm'
SRC_WRITE_FILE='/home/christine/news_micro_2/newstest2015-ende-src.en'
REF_WRITE_FILE='/home/christine/news_micro_2/newstest2015-ende-ref.de'
DOC_TEXT_FILE='/home/christine/news_micro_2/newstest2015_doc_text.h5'
TEST_SENT_DOC_ALIGN='/home/christine/news_micro_2/newstest2015_sen_doc_alignment.h5'

python3 test_dev_prepare.py --file_read_src $SRC_FILE --file_read_ref $REF_FILE \
--file_write_src $SRC_WRITE_FILE --file_write_ref $REF_WRITE_FILE \
--doc_text_save  $DOC_TEXT_FILE --sen_doc_align $TEST_SENT_DOC_ALIGN

#gathering all dictionaries in one
ALL_SENT_DOC_ALIGN='/home/christine/news_micro_2/all_sen_doc_alignment.h5'
python3 prepare_sen_doc_dict.py  --dict_training_path $TRAIN_SENT_DOC_ALIGN  --dict_valid_path $VALID_SENT_DOC_ALIGN \
--dict_test_path  $TEST_SENT_DOC_ALIGN --dict_sent_doc_all_path $ALL_SENT_DOC_ALIGN







#!/bin/bash -v


# suffix of source language files
SRC=en

# suffix of target language files
TRG=de

# number of merge operations
bpe_operations=32000

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder='../mosesdecoder'

# path to subword segmentation scripts: https://github.com/rsennrich/subword-nmt
data_folder='/home/christine/news_micro_2/data'
model_folder='/home/christine/news_micro_2/model'
# tokenize
for prefix in train valid test2015
do
    cat $data_folder/$prefix.$SRC \
        | $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $SRC \
        | $mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l $SRC > $data_folder/$prefix.tok.$SRC

    test -f $data_folder/$prefix.$TRG || continue

    cat $data_folder/$prefix.$TRG \
        | $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $TRG \
        | $mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l $TRG > $data_folder/$prefix.tok.$TRG
done


# train truecaser
$mosesdecoder/scripts/recaser/train-truecaser.perl -corpus $data_folder/train.tok.$SRC -model $model_folder/tc.$SRC
$mosesdecoder/scripts/recaser/train-truecaser.perl -corpus $data_folder/train.tok.$TRG -model $model_folder/tc.$TRG

# apply truecaser (cleaned training corpus)
for prefix in train valid test2015
do
    $mosesdecoder/scripts/recaser/truecase.perl -model $model_folder/tc.$SRC < $data_folder/$prefix.tok.$SRC > $data_folder/$prefix.tok.tc.$SRC
    #test -f data/$prefix.tok.$TRG || continue
    $mosesdecoder/scripts/recaser/truecase.perl -model $model_folder/tc.$TRG < $data_folder/$prefix.tok.$TRG > $data_folder/$prefix.tok.tc.$TRG
done
#adding <DOC> token
for prefix in train valid test2015
do
  sed -i 's/^/<DOC> /' $data_folder/$prefix.tok.tc.$SRC
done

#!/bin/bash -v

# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh
# suffix of source language files
SRC=en

# suffix of target language files
TRG=es
#to change if we do not need the doc token
needs_doc_token=true
#if we need to train teh trucase or just work with what we have
needs_truecasing=true
# number of merge operations
bpe_operations=32000

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder='../mosesdecoder'

# path to subword segmentation scripts: https://github.com/rsennrich/subword-nmt
data_folder='/home/usuaris/scratch/christine.raouf.saad/google_dataset/data'
#to train tue caser
train_data_folder='/home/usuaris/scratch/christine.raouf.saad/news_v11/NCv11_en-es/data'
train_file='news_all'
model_folder='/home/christine/news_micro_2/model'
# normalize, punctuate, tokenize
for prefix in generate
do
    cat $data_folder/$prefix.$SRC \
        | $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $SRC \
        | $mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l $SRC > $data_folder/$prefix.tok.$SRC

    test -f $data_folder/$prefix.$TRG || continue

    cat $data_folder/$prefix.$TRG \
        | $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $TRG \
        | $mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l $TRG > $data_folder/$prefix.tok.$TRG
done


# train truecaser..we have trained one
if $needs_truecasing
then
  $mosesdecoder/scripts/recaser/train-truecaser.perl -corpus $train_data_folder/$train_file.tok.$SRC -model $train_data_folder/tc.$SRC
  $mosesdecoder/scripts/recaser/train-truecaser.perl -corpus $train_data_folder/$train_file.tok.$TRG -model $train_data_folder/tc.$TRG
fi

# apply truecaser (cleaned training corpus)
for prefix in   generate
do
    $mosesdecoder/scripts/recaser/truecase.perl -model $train_data_folder/tc.$SRC < $data_folder/$prefix.tok.$SRC > $data_folder/$prefix.tok.tc.$SRC
    #test -f data/$prefix.tok.$TRG || continue
    $mosesdecoder/scripts/recaser/truecase.perl -model $train_data_folder/tc.$TRG < $data_folder/$prefix.tok.$TRG > $data_folder/$prefix.tok.tc.$TRG
done
#adding <DOC> token
failed=true
if $needs_doc_token
then
    for prefix in generate
    do
      sed 's/^/<DOC> /' $data_folder/$prefix.tok.tc.$SRC > $prefix.tok.tc.doc.$SRC
    done
else
   echo "no need to add <doc> for baseline"
fi


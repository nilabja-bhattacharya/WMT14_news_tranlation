 
#  echo tokenization
#  ~/smt/mosesdecoder/scripts/tokenizer/tokenizer.perl -l en \
#     < ~/smt/temp/corpus/training/hindien-train.en   \
#     > ~/smt/temp/corpus/hindien.tok.en
#  ~/smt/mosesdecoder/scripts/tokenizer/tokenizer.perl -l hi \
#     < ~/smt/temp/corpus/training/hindien-train.hi    \
#     > ~/smt/temp/corpus/hindien.tok.hi

# echo truecaser training
#  ~/smt/mosesdecoder/scripts/recaser/train-truecaser.perl \
#      --model ~/smt/temp/corpus/truecase-model.en --corpus     \
#      ~/smt/temp/corpus/hindien.tok.en
#  ~/smt/mosesdecoder/scripts/recaser/train-truecaser.perl \
#      --model ~/smt/temp/corpus/truecase-model.hi --corpus     \
#      ~/smt/temp/corpus/hindien.tok.hi


# echo truecasing
#  ~/smt/mosesdecoder/scripts/recaser/truecase.perl \
#    --model ~/smt/temp/corpus/truecase-model.en         \
#    < ~/smt/temp/corpus/hindien.tok.en \
#    > ~/smt/temp/corpus/hindien.true.en
#  ~/smt/mosesdecoder/scripts/recaser/truecase.perl \
#    --model ~/smt/temp/corpus/truecase-model.hi         \
#    < ~/smt/temp/corpus/hindien.tok.hi \
#    > ~/smt/temp/corpus/hindien.true.hi

# echo cleaning to length 100
#    ~/smt/mosesdecoder/scripts/training/clean-corpus-n.perl \
#     ~/smt/temp/corpus/hindien.true hi en \
#     ~/smt/temp/corpus/hindien.clean 1 80


# echo language modeling
# #mkdir ~/lm
# ~/smt/mosesdecoder/bin/lmplz -o 3 -S 70% <~/smt/temp/corpus/hindien.true.en > ~/smt/temp/lm/hindien.arpa.en

# ~/smt/mosesdecoder/bin/build_binary \
#    ~/smt/temp/lm/hindien.arpa.en \
#    ~/smt/temp/lm/hindien.blm.en


# # training translation system
# echo training
# #mkdir ~/working
#  nohup nice ~/smt/mosesdecoder/scripts/training/train-model.perl -root-dir train \
#  -corpus ~/smt/temp/corpus/hindien.clean                             \
#  -f hi -e en -alignment grow-diag-final-and -reordering msd-bidirectional-fe \
#  -lm 0:3:$HOME/smt/temp/lm/hindien.blm.en:8     -cores 4                     \
#  -external-bin-dir ~/smt/mosesdecoder/tools >& ~/smt/temp/working/training.out &

# wait

#   ~/smt/mosesdecoder/scripts/tokenizer/tokenizer.perl -l en \
#     < ~/smt/temp/corpus/dev/hindien-dev.en > ~/smt/temp/corpus/hindien-dev.tok.en
#   ~/smt/mosesdecoder/scripts/tokenizer/tokenizer.perl -l hi \
#     < ~/smt/temp/corpus/dev/hindien-dev.hi > ~/smt/temp/corpus/hindien-dev.tok.hi
#   ~/smt/mosesdecoder/scripts/recaser/truecase.perl --model ~/smt/temp/corpus/truecase-model.en \
#     < ~/smt/temp/corpus/hindien-dev.tok.en > ~/smt/temp/corpus/hindien-dev.true.en
#   ~/smt/mosesdecoder/scripts/recaser/truecase.perl --model ~/smt/temp/corpus/truecase-model.hi \
#     < ~/smt/temp/corpus/hindien-dev.tok.hi > ~/smt/temp/corpus/hindien-dev.true.hi

# #tuning
# echo tuning
# nohup nice ~/smt/mosesdecoder/scripts/training/mert-moses.pl \
#   ~/smt/temp/corpus/hindien-dev.true.hi ~/smt/temp/corpus/hindien-dev.true.en \
#   ~/smt/mosesdecoder/bin/moses train/model/moses.ini --decoder-flags="-threads 4"   --mertdir ~/smt/mosesdecoder/bin/ \
#   &> ~/smt/temp/working/mert.out &

# wait
# # mkdir ~/working/binarised-model
# #  cd ~/working
#  ~/smt/mosesdecoder/bin/processPhraseTableMin \
#    -in ~/smt/temp/train/model/phrase-table.gz -nscores 4 \
#    -out ~/smt/temp/working/binarised-model/phrase-table
#  ~/smt/mosesdecoder/bin/processLexicalTableMin \
#    -in ~/smt/temp/train/model/reordering-table.wbe-msd-bidirectional-fe.gz \
#    -out ~/smt/temp/working/binarised-model/reordering-table


# echo testing
# cd ~/corpus
#  ~/smt/mosesdecoder/scripts/tokenizer/tokenizer.perl -l en \
#    < ~/smt/temp/corpus/test/hindien-test.en > ~/smt/temp/corpus/hindien-test.tok.en
#  ~/smt/mosesdecoder/scripts/tokenizer/tokenizer.perl -l hi \
#    < ~/smt/temp/corpus/test/hindien-test.hi > ~/smt/temp/corpus/hindien-test.tok.hi
#  ~/smt/mosesdecoder/scripts/recaser/truecase.perl --model ~/smt/temp/corpus/truecase-model.en \
#    < ~/smt/temp/corpus/hindien-test.tok.en > ~/smt/temp/corpus/hindien-test.true.en
#  ~/smt/mosesdecoder/scripts/recaser/truecase.perl --model ~/smt/temp/corpus/truecase-model.hi \
#    < ~/smt/temp/corpus/hindien-test.tok.hi > ~/smt/temp/corpus/hindien-test.true.hi

# #  cd ~/working
#  ~/smt/mosesdecoder/scripts/training/filter-model-given-input.pl             \
#    ~/smt/temp/working/filtered-test ~/smt/temp/mert-work/moses.ini ~/smt/temp/corpus/hindien-test.true.hi \
#    -Binarizer ~/smt/mosesdecoder/bin/processPhraseTableMin

# nohup nice ~/smt/mosesdecoder/bin/moses            \
#    -f ~/smt/temp/working/filtered-test/moses.ini   \
#    < ~/smt/temp/corpus/hindien-test.true.hi                \
#    > ~/smt/temp/working/hindien-test.translated.en         \
#    2> ~/smt/temp/working/hindien.out 
#   wait
echo bleu_score
 ~/smt/mosesdecoder/scripts/generic/multi-bleu.perl \
   -lc ~/smt/temp/corpus/hindien-test.true.en              \
   < ~/smt/temp/working/hindien-test.translated.en

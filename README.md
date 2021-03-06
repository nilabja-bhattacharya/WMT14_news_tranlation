# WMT14 News Tranlation on hindi english data corpus
In this project our task was to perform Machine Translation on a English-Hindi News Corpus. We have solved it using SMT and NMT and we presented the BLEU score on each system 

Keywords: SMT, NMT, General, Concat, Dot, Attention


1.  SMT: Statistical machine translation is a machine translation
    paradigm where translations are generated on the basis of
    statistical models whose parameters are derived from the analysis of
    bilingual text corpora. The statistical approach contrasts with the
    rule-based approaches to machine translation as well as with example
    based machine translation.\

2.  NMT: Neural Machine Translation has been re- ceiving considerable
    attention in recent years, given its superior performance without
    the de- mand of heavily hand crafted engineering ef- forts. NMT
    often outperforms Statistical Machine Translation (SMT) techniques
    but it still struggles if the parallel data is insufficient like in
    the case of Indian languages.

3.  Attention: The basic idea: each time the model predicts an output
    word, it only uses parts of an input where the most relevant
    information is concentrated instead of an entire sentence. In other
    words, it only pays attention to some input words

4.  MOSES: Moses is a free software,SMT engine that can be used to train
    statistical models of text translation from a source language to a
    target language. Moses then allows new source-language text to be
    decoded using these models to produce automatic translations in the
    target language. Training requires a parallel corpus of passages in
    the two languages, typically manually translated sentence pairs.

5.  Language Modeling: P(e) Before finding P(f|e) we need to build a
    machine that assigns a probability P(e) to each English sentence e.
    This is called a language model.

6.  N-grams: For computers, the easiest way to break a string down into
    components is to consider substrings. An n-word substring is called
    an n-gram. If n=2, we say bigram. If n=3, we say trigram. If n=1,
    nerds say unigram, and normal people say word.

7.  Translation Modeling: P(f|e), the probability of a string f
    given an English string e. This is called a translation model. P(f|e) will be a module in overall f-to-e machine translation.
    When we see a string f, what we need to consider for e is that how
    likely it is to be uttered, and likely to subsequently translate to
    f? We’re looking for the e that maximizes P(e) * P(f | e).

8.  Alignment Probabilities: For a given sentence pair: what is the
    probability of the words being aligned in particular arrangement.
    For a given sentence pair, the probabilities of the various possible
    alignments should add to one. P(a | e,f) = P(a,f | e) / P(f | e)
    (f | e) = sum(P(a,f | e))

9.  Expectation Maximization Algorithm

    1.  Assign uniform probability values for the alignments.

    2.  From this we get the “expected counts” of alignments.

    3.  From these expected counts we get the “revised” probabilities.

    4.  Iterate steps 2 and 3 until convergence
     
### Steps to use it
    1. To use smt_mosses.sh run the script by setting the path relative to ~/smt/temp/
    2. To generate translated sentence run the test.sh script by setting path relative to ~/smt/temp/
    3. We can check the bleu score of smt model using the helper.ipynb file in smt directory
    4. There is individual python script for each model in nmt directory which can be run using - python <python_script.py> 
    5. We can check the performance using the checkpoints created in step 3, in colab by uploading the checkpoints and running the notebook. To use a checkpoint created in a particular epoch, set - itr - in notebook to the particular epoch number. 
    6. To use the checkpoints that I've created, please add shortcut to the drive in your google drive and then run the notebook

### LINKS

-   [Slides](https://docs.google.com/presentation/d/1tylPZVzRy1UaASTTlmhpExfjZEGoUNs8RqHTxwDxbX8/edit?usp=sharing)

-   [Github](https://github.com/nilabja-bhattacharya/WMT14_news_tranlation)

-   [Complete
    Repo](https://drive.google.com/open?id=1XLBd0VSe3Kx7ql8VjNYAB23YbxoPXcts)


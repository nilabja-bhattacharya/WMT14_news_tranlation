# WMT14_news_tranlation
In this project our task was to perform Machine Translation on a
English-Hindi News Corpus. We have solved it using

1.  SMT: We built a MOSES phrase based Statistical Machine Translation
    system to translate the test sets in the IIT Bombay dataset for the
    WMT News Translation task.\

2.  NMT: Neural Machine Translation. We built a Sequence to Sequence
    network or Encoder-Decoder network, a model consisting of two RNNs
    called the encoder and decoder.

SMT, NMT, General, Concat, Dot, Attention

INTRODUCTION
============

1.  SMT:Statistical machine translation is a machine translation
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
    the case of Indian languages.\

3.  Attention: The basic idea: each time the model predicts an output
    word, it only uses parts of an input where the most relevant
    information is concentrated instead of an entire sentence. In other
    words, it only pays attention to some input words\

4.  MOSES: Moses is a free software,SMT engine that can be used to train
    statistical models of text translation from a source language to a
    target language. Moses then allows new source-language text to be
    decoded using these models to produce automatic translations in the
    target language. Training requires a parallel corpus of passages in
    the two languages, typically manually translated sentence pairs.\

5.  Language Modeling: $$P(e)$$ Before finding p(f|e) we need to build a
    machine that assigns a probability P(e) to each English sentence e.
    This is called a language model.\

6.  N-grams: For computers, the easiest way to break a string down into
    components is to consider substrings. An n-word substring is called
    an n-gram. If n=2, we say bigram. If n=3, we say trigram. If n=1,
    nerds say unigram, and normal people say word.\

7.  Translation Modeling: $$P(f | e)$$, the probability of a string f
    given an English string e. This is called a translation model. P(f |
    e) will be a module in overall f-to-e machine translation.\
    When we see a string f, what we need to consider for e is that how
    likely it is to be uttered, and likely to subsequently translate to
    f? We’re looking for the e that maximizes $$P(e) * P(f | e)$$.\

8.  Alignment Probabilities: For a given sentence pair: what is the
    probability of the words being aligned in particular arrangement.
    For a given sentence pair, the probabilities of the various possible
    alignments should add to one. $$P(a | e,f) = P(a,f | e) / P(f | e)$$
    $$P(f | e) = \sum P(a,f | e)$$\

9.  Expectation Maximization Algorithm

    1.  Assign uniform probability values for the alignments.

    2.  From this we get the “expected counts” of alignments.

    3.  From these expected counts we get the “revised” probabilities.

    4.  Iterate steps 2 and 3 until convergence

LITERATURE SURVEY
=================

These are the state of the art papers in this area that we referenced.

​1. “Neural machine translation by jointly learning to align and
translate” (Dzmitry Bahdanau,KyungHyun Cho) In this paper the authors
conjecture that the use of a fixed-length vector is a bottleneck in
improving the performance of this basic encoder–decoder architecture,
and propose to extend this by allowing a model to automatically
soft-search for parts of a source sentence that are relevant to
predicting a target word, without having to form these parts as a hard
segment explicitly. With this new approach, they achieve a translation
performance comparable to the existing state-of-the-art phrase-based
system on the task of English-to-French translation. In order to address
this issue that the performance of a basic encoder–decoder deteriorates
rapidly as the length of an input sentence increases, an extension to
the encoder–decoder model which learns to align and translate jointly is
suggested. Each time the proposed model generates a word in a
translation, it (soft-)searches for a set of positions in a source
sentence where the most relevant information is concentrated. The model
then predicts a target word based on the context vectors associated with
these source positions and all the previous generated target words.\

​2. Machine Translation with parfda, Moses, kenlm, nplm, and PRO (Ergun
Bicici) In this paper they build parfda (parallel feature weight decay
algorithms) Moses SMT models for most language pairs in the news
translation task. The authors experiment with a hybrid approach using
neural language models integrated into Moses. They obtain the
constrained data statistics on the machine translation task, the
coverage of the test sets, and the upper bounds on the translation
results. Parfda parallelize feature decay algorithms (FDA), a class of
instance selection algorithms that decay feature weights, for fast
deployment of accurate SMT systems. They train 6-gram LM using kenlm and
use mgiza for word alignment.

Research Methods
================

-   Dataset\
    \
    We have used English-Hindi the parallel training data which consists
    of the new HindEnCorp, collected by Charles University, linked in
    the WMT-2014 translation task. The English-Hindi corpus contains
    parallel corpus for English-Hindi of around 2.7 lakh sentences.\

-   Data Preprocessing\
    \
    We used Moses-toolkit for tokenization and cleaning the English side
    of the data. The Hindi side of the data is first normalized with
    Indic NLP library1 followed by tokenization with the same library.
    As our preprocessing step, we re- moved all the sentences of length
    greater than 80 from our training corpus.\

-   Architecture: SMT\
    \
    We used a hybrid approach using neural language models integrated
    into Moses. Obtained the constrained data statistics on the machine
    translation task, the coverage of the test sets, and the upper
    bounds on the translation results.\
    Then trained 3-gram LM using kenlm. In phrase-based translation, the
    aim is to reduce the restrictions of word-based translation by
    translating whole sequences of words, where the lengths may differ.
    The sequences of words are called blocks or phrases, but typically
    are not linguistic phrases, but pharasemes found using statistical
    methods from corpora.\
    The chosen phrases are further mapped one-to-one based on a phrase
    translation table, and may be reordered. This table can be learnt
    based on word-alignment, or directly from a parallel corpus. The
    second model is trained using the expectation maximization
    algorithm, similarly to the word-based IBM Model.
    ![image](eleven.png) ![image](fourteen.png) Score is computed
    incrementally for each partial hypothesis.\
    Components :\

    1.  Phrase translation : Picking phrase f ̄ to be translated as a
        phrase e ̄ Look up score$$\phi(f ̄|e ̄\ )$$from phrase translation
        table.

    2.  Reordering : Previous phrase ended in end(i−1), current phrase
        starts at start(i). Compute $$d(start(i) − end(i−1) − 1)$$

    \

-   Training Details: SMT\
    \
    Corpus Preparation\
    To prepare the data for training the translation system, we have to
    perform the following steps:

    -   Tokenization:\
        This means that spaces have to be inserted between (e.g.) words
        and punctuation.\

    -   Truecasing:\
        The initial words in each sentence are converted to their most
        probable casing. This helps reduce data sparsity.\

    -   Cleaning:\
        Long sentences and empty sentences are removed as they can cause
        problems with the training pipeline, and obviously mis-aligned
        sentences are removed.\

    Language Model Training\
    \
    The language model (LM) is used to ensure fluent output, so it is
    built with the target language (i.e English in this case).\
    \
    Training the Translation System\
    \
    For training we run word-alignment (using GIZA++), phrase extraction
    and scoring, create lexicalized reordering tables and create your
    Moses configuration file.\
    \
    Tuning\
    \
    Tuning refers to the process of finding the optimal weights for this
    linear model, where optimal weights are those which maximise
    translation performance on a small set of parallel sentences (the
    tuning set). Translation performance is usually measured with Bleu,
    but the tuning algorithms all support (at least in principle) the
    use of other performance measures.\

    Architecture: NMT\
    \
    We are using the attention based encoder-decoder architecture. The
    NMT model consists of an encoder and a decoder, each of which is a
    Recurrent Neural Network. (RNN)\
    An encoder neural network reads and encodes a source sen- tence into
    a fixed-length vector. A decoder then outputs a translation from the
    encoded vector. The whole encoder–decoder system, which consists of
    the encoder and the decoder for a language pair, is jointly trained
    to maximize the probability of a correct translation given a source
    sentence.\
    From a probabilistic perspective, translation is equivalent to
    finding a target sentence y that max- imizes the conditional
    probability of y given a source sentence x,
    i.e.,$$arg maxy  p(y | x)$$

-   Training details: NMT\
    \

    ![image](one.png) Data Processing\
    We convert the data into one-hot encoding format and then learn the
    embedding for each word during training.\
    ![image](two.png)\
    \
    In NMT model we have used biLSTM network with layers 2 and 256
    hidden unit per biLSTM unit in encoder and decoder. For decoder we
    have used various type of attention to improve performance\
    \
    Enoder Unit\
    \
    We send the hindi sentence through encoder to get a vector space of
    the sentence. Then this output is passed through the decoder unit to
    get output.\
    \
    ![image](three.png)\
    \
    Decoder Unit\
    \
    Simple Decoder\
    In the simplest seq2seq decoder we use only last output of the
    encoder. This last output is sometimes called the context vector as
    it encodes context from the entire sequence. This context vector is
    used as the initial hidden state of the decoder.\
    \
    ![image](four.png)\
    \
    Attention Decoder\
    \
    Attention allows the decoder network to “focus” on a different part
    of the encoder’s outputs for every step of the decoder’s own
    outputs. First we calculate a set of attention weights. These will
    be multiplied by the encoder output vectors to create a weighted
    combination.\
    \
    \

FINDINGS AND ANALYSIS
=====================

[H]

<span>X|X</span> System & BLEU\
SMT & 0.2540\
Sequence to sequence & 0.2577\
Concat Attention & 0.2492\
Dot Attention & 0.2574\
General Attention & 0.2613\

[experimental setup]

![Plot of loss vs iteration](seq2seq.png "fig:") [fig:1]

![Plot of loss vs iteration](concat.png "fig:") [fig:1]

![Plot of loss vs iteration](dot.png "fig:") [fig:1]

![Plot of loss vs iteration](general.png "fig:") [fig:1]

### Analysis

-   SMT system sentence were not that logical and we had to interpret
    the translation

-   We had to deal with sparsity problem in SMT system

-   In sequence to sequence model, the sentence length of generated
    sentence went upto max length and usually consisted of repeated
    words

-   General attention based NMT system performed best among all other
    model

CONCLUSIONS
===========

General attention based NMT system performed better as compared to all
other models. The BLEU score can be further improved using a transformer
model and have large dataset. We can also use coverage to make a better
model.

LINKS
=====

-   [Slides](https://docs.google.com/presentation/d/1tylPZVzRy1UaASTTlmhpExfjZEGoUNs8RqHTxwDxbX8/edit?usp=sharing)

-   [Github](https://github.com/nilabja-bhattacharya/WMT14_news_tranlation)

-   [Complete
    Repo](https://drive.google.com/open?id=1XLBd0VSe3Kx7ql8VjNYAB23YbxoPXcts)

<span>99</span>

https://pytorch.org/tutorials/
http://www.statmt.org/moses/?n=Moses.Baseline Machine Translation with
parfda, Moses, kenlm, nplm, and PRO(Ergun Bicici) “Neural machine
translation by jointly learning to align and translate” (Dzmitry
Bahdanau,KyungHyun Cho)
All the work related to this can be found here -> [link to WMT news translation!](https://drive.google.com/open?id=1XLBd0VSe3Kx7ql8VjNYAB23YbxoPXcts)

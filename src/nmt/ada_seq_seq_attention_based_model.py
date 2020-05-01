# -*- coding: utf-8 -*-
"""Ada_seq_seq_attention based model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hs2EbMNktX4lkVE-1emSIh6YL7Y_PN0a
"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

from google.colab import drive
drive.mount('/gdrive', force_remount=True)

path = ""

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""### Language Model from input and output sentences"""

SOS_token = 0
EOS_token = 1
UNK_token = 2


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2:"UNK"}
        self.n_words = 3  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
#     s = re.sub(r"([.!?|])", r" \1", s)
    #s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
#     lines = open('%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
#         read().strip().split('\n')
    lines1 = open(path+"corpus/training/hindien-train.tok."+lang1).readlines()
    lines2 = open(path+"corpus/training/hindien-train.tok."+lang2).readlines()
    #print(lines)
    # Split every line into pairs and normalize
    pairs = [[normalizeString(x), normalizeString(y)] for (x,y) in zip(lines1, lines2)]
#     pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
#     print(pairs[:50])
    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

MAX_LENGTH = 100

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=False):    
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    #print(input_lang, output_lang, pairs)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepareData('en', 'hi', True)
#print(input_lang, output_lang, pairs)
print(random.choice(pairs))

"""## Encoder"""

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        #changes
        self.gru = nn.LSTM(hidden_size, hidden_size, num_layers, bidirectional=True)

    def forward(self, input, hidden, cell):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, (hidden, cell) = self.gru(output, (hidden, cell))
        return output, hidden, cell

    def initHidden(self):
        return torch.zeros(self.num_layers*2, 1, self.hidden_size, device=device)

"""## Decoder"""

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        #changes
        self.gru = nn.LSTM(hidden_size, hidden_size, num_layers, bidirectional=True)
        self.out = nn.Linear(hidden_size*2, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, cell):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, (hidden, cell) = self.gru(output, (hidden, cell))
        output = self.softmax(self.out(output[0]))
        return output, hidden, cell

    def initHidden(self):
        return torch.zeros(self.num_layers*2, 1, self.hidden_size, device=device)

"""## Attention Based Decoder"""

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers,  dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.num_layers = num_layers
        self.embedding = nn.Embedding(self.output_size, self.hidden_size*2)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size *4 , self.hidden_size)
        self.attn_general = nn.Linear(self.max_length, self.max_length)
        self.attn_concat = nn.Linear(self.hidden_size*3, self.max_length)
        self.lcl_wa_into_hs = nn.Linear(self.hidden_size*2,self.hidden_size*2)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers, bidirectional=True)
        self.out = nn.Linear(self.hidden_size*2, self.output_size)

    def forward(self, input, hidden, cell, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        #print(embedded.size(), hidden.size())
        # attn_weights = F.softmax(
        #     self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        X = None
        #print(embedded.shape, encoder_outputs.shape)
        if attention_type == 'dot':
            X = torch.matmul(embedded[0], encoder_outputs.T) #dot product
        elif attention_type == 'general':
            X = self.attn_general(torch.matmul(self.lcl_wa_into_hs(embedded[0]), encoder_outputs.T)) ## general
        elif attention_type == 'concat':
            X = self.attn_concat(torch.cat((embedded[0], hidden[0]), 1))  #concat
        attn_weights = F.softmax(X, dim = 1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        #print(attn_applied.shape)
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, (hidden, cell) = self.gru(output, (hidden, cell))

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, cell, attn_weights

    def initHidden(self):
        return torch.zeros(self.num_layers*2, 1, self.hidden_size, device=device)

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] if word in lang.word2index.keys() else UNK_token for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    #print(indexes)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

"""## Training of Neural Network"""

teacher_forcing_ratio = 0.5
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()
    encoder_cell = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size*2, device=device)

    loss = 0

    for ei in range(input_length):
        #print(ei)
        encoder_output, encoder_hidden, encoder_cell = encoder(
            input_tensor[ei], encoder_hidden, encoder_cell)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden
    decoder_cell = encoder_cell
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            if attention == True:
                decoder_output, decoder_hidden, decoder_cell, decoder_attention = decoder(
                    decoder_input, decoder_hidden, decoder_cell, encoder_outputs)
            else:
                decoder_output, decoder_hidden, decoder_cell = decoder(
                    decoder_input, decoder_hidden, decoder_cell)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            if attention == True:
                decoder_output, decoder_hidden, decoder_cell, decoder_attention = decoder(
                    decoder_input, decoder_hidden, decoder_cell, encoder_outputs)
            else:
                decoder_output, decoder_hidden, decoder_cell = decoder(
                    decoder_input, decoder_hidden, decoder_cell)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

"""## Gradient Descent"""

def trainIters(encoder, decoder, n_iters, print_every=1000, learning_rate=0.001):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    # lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate,  betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
    if from_checkpoint == True:
      checkpoint_encoder = torch.load(path+"/checkpoint/encoder_checkpoint/"+str(itr)+".pth")
      encoder.load_state_dict(checkpoint_encoder['model_state_dict'])
      encoder_optimizer.load_state_dict(checkpoint_encoder['optimizer_state_dict'])
      checkpoint_decoder = torch.load(path+"/checkpoint/decoder_checkpoint/"+str(itr)+".pth")
      decoder.load_state_dict(checkpoint_decoder['model_state_dict'])
      decoder_optimizer.load_state_dict(checkpoint_decoder['optimizer_state_dict'])
      epoch = checkpoint_encoder['epoch']
    
    criterion = nn.NLLLoss()

    for iter in range(itr+1, itr + n_iters + 1):
        print("Epoch:{0}".format(iter))
        training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(len(pairs))]
        for b in range(1,len(pairs)+1):
              training_pair = training_pairs[b-1]
              input_tensor = training_pair[0]
              target_tensor = training_pair[1]

              loss = train(input_tensor, target_tensor, encoder,
                          decoder, encoder_optimizer, decoder_optimizer, criterion)
              print_loss_total += loss
              plot_loss_total += loss
        
              if b % print_every == 0:
                  print_loss_avg = print_loss_total / (print_every)
                  print_loss_total = 0
                  plot_losses.append(print_loss_avg)
                  print('%s (%d %d%%) %.4f' % (timeSince(start, b / len(pairs)),
                                              b , b / len(pairs) * 100, print_loss_avg))

        # if iter % print_every == 0:
        #     print_loss_avg = print_loss_total / (print_every*len(pairs))
        #     print_loss_total = 0
        #     plot_losses.append(print_loss_avg)
        #     print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
        #                                  iter, iter / n_iters * 100, print_loss_avg))
        
        torch.save({
        'epoch': iter,
        'model_state_dict': encoder.state_dict(),
        'optimizer_state_dict': encoder_optimizer.state_dict(),
        }, path+"/checkpoint/encoder_checkpoint/"+str(iter)+".pth")
        torch.save({
        'epoch': iter,
        'model_state_dict': decoder.state_dict(),
        'optimizer_state_dict': decoder_optimizer.state_dict(),
        }, path+"/checkpoint/decoder_checkpoint/"+str(iter)+".pth")

    #showPlot(plot_losses)
    return plot_losses

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

"""## Evaluation of test data"""

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        encoder_cell = encoder.initHidden()
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size*2, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden, encoder_cell = encoder(input_tensor[ei],
                                                     encoder_hidden, encoder_cell)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            if attention == True:
                decoder_output, decoder_hidden, decoder_cell, decoder_attention = decoder(
                    decoder_input, decoder_hidden, decoder_cell, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
            else:
                decoder_output, decoder_hidden, decocer_cell = decoder(
                    decoder_input, decoder_hidden, decoder_cell)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == UNK_token:
                decoded_words.append('<UNK>')
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

from nltk.translate import bleu
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

def calculate_bleu(pred_trg, real_trg):   
    smoothie = SmoothingFunction().method4
    return sentence_bleu(real_trg, pred_trg, smoothing_function=smoothie)
    # smoothie = SmoothingFunction().method4
    # return bleu(real_trg, pred_trg, smoothing_function=smoothie)
def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
        print(calculate_bleu(pair[1], output_words))

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy()[:20][:20], cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))

    plt.show()

def evaluateAndShowAttention(encoder, decoder, input_sentence, real_output_sentence):
    output_words, attentions = evaluate(
        encoder, decoder, input_sentence)
    output_sentence = ' '.join(output_words)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    print('>', input_sentence)
    print('=', real_output_sentence)
    print('<', output_sentence)
    blu = calculate_bleu(real_output_sentence, output_words)
    print('')
    # if attention == True:
    #   showAttention(input_sentence, output_words, attentions)
    return blu


def evaluate_on_test(encoder, decoder):
  trg = open(path+"corpus/hindien-dev.tok.en").readlines()
  real_out = open(path+"corpus/hindien-dev.tok.en").readlines()
  blu = 0
  for t, r in zip(trg[:100], real_out[:100]):
      t = t.replace('\n','')
      r = r.replace('\n', '')
      if len(t.split(' ')) < MAX_LENGTH and len(r.split(' ')) < MAX_LENGTH:
          blu += evaluateAndShowAttention(encoder, decoder, t, r)
  return blu/len(trg)

"""## Paper 1 implementation:

Here I'm using a sequence to sequence encoder decoder model of GRU with 256 hidden units.

Encoder converts the input sequences to a normalized vector space and then decoder unit interpret the output sentence from this normalized vector space.

I'm running the code for 50000 iterations.
"""

from_checkpoint = False
save_checkpoint = True
itr = 0

hidden_size = 256
num_layers = 2
attention = False
torch.cuda.empty_cache() 
encoder1 = EncoderRNN(input_lang.n_words, hidden_size, num_layers).to(device)
decoder1 = DecoderRNN(hidden_size, output_lang.n_words, num_layers).to(device)
losses = trainIters(encoder1, decoder1, 3, print_every=1000)

showPlot(losses)

if from_checkpoint == True:
      checkpoint_encoder = torch.load(path+"/checkpoint/encoder_checkpoint/"+str(itr)+".pth")
      encoder1.load_state_dict(checkpoint_encoder['model_state_dict'])
      checkpoint_decoder = torch.load(path+"/checkpoint/decoder_checkpoint/"+str(itr)+".pth")
      decoder1.load_state_dict(checkpoint_decoder['model_state_dict'])

blu = evaluateRandomly(encoder1, decoder1)
print(blu)

blu = evaluate_on_test(encoder1, decoder1)
print(blu)

"""## Attention Based Models

### DOT variant
"""

hidden_size = 256
num_layers = 1
attention = True
attention_type = 'dot'
encoder1 = EncoderRNN(input_lang.n_words, hidden_size, num_layers).to(device)
decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, num_layers, dropout_p=0.1).to(device)
losses = trainIters(encoder1, decoder1, 2, print_every=1)

showPlot(losses)

evaluateRandomly(encoder1, decoder1)

blu = evaluate_on_test(encoder1, decoder1)
print(blu)

"""### General Variant"""

hidden_size = 1
num_layers = 2
attention = True
attention_type = 'general'
encoder1 = EncoderRNN(input_lang.n_words, hidden_size, num_layers).to(device)
decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, num_layers, dropout_p=0.1).to(device)
losses = trainIters(encoder1, decoder1, 100000, print_every=10)

showPlot(losses)

evaluateRandomly(encoder1, decoder1)

blu = evaluate_on_test(encoder1, decoder1)
print(blu)

"""### Concat Varient"""

hidden_size = 256
num_layers = 1
attention_type = 'concat'
attention = True
encoder1 = EncoderRNN(input_lang.n_words, hidden_size, num_layers).to(device)
decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, num_layers, dropout_p=0.1).to(device)
losses = trainIters(encoder1, decoder1, 10, print_every=1)

showPlot(losses)

evaluateRandomly(encoder1, decoder1)

blu = evaluate_on_test(encoder1, decoder1)
print(blu)


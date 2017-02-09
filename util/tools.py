# coding=utf-8
import cPickle
import re
import math
import jieba
import numpy as np
from collections import defaultdict
import sys
from pymongo import MongoClient

reload(sys)
sys.setdefaultencoding("utf-8")

def preprocess(text):
    """
    Preprocess text for encoder
    """
    X = []
    # sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    for t in text:
        # sents = sent_detector.tokenize(t)
        # result = ''
        # for s in sents:
        #     tokens = word_tokenize(s)
        #     result += ' ' + ' '.join(tokens)
        X.append(t)
    return X


def load_data(loc='../news_data/'):
    """
    Load the TREC question-type dataset
    """
    train, test = [], []
    with open(loc + 'train.label', 'rb') as f:
        for line in f:
            train.append(line.strip())
    with open(loc + 'text_preprocess.label', 'rb') as f:
        for line in f:
            test.append(line.strip())
    return train, test

def prepare_data(text):
    """
    Prepare data
    """
    labels = [t.split()[0] for t in text]
    labels = [l.split(':')[0] for l in labels]
    X = [t.split()[1:] for t in text]
    X = [' '.join(t) for t in X]
    return X, labels

def prepare_labels(labels):
    """
    Process labels to numerical values
    """
    d = {}
    count = 0
    setlabels = set(labels)
    for w in setlabels:
        d[w] = count
        count += 1
    idxlabels = np.array([d[w] for w in labels])
    return idxlabels

def build_vocab(text):

    vocab = defaultdict(float)
    for sent in text:
        words = sent.split()
        for word in words:
            vocab[word] += 1

    wordtoix = defaultdict(float)
    ixtoword = defaultdict(float)

    count = 0
    for w in vocab.keys():
        wordtoix[w] = count
        ixtoword[count] = w
        count += 1

    return wordtoix, ixtoword

def get_idx_from_sent(sent, wordtoix):
    """
    Transforms sentence into a list of indices.
    """
    x = []
    words = sent.split()
    for word in words:
        x.append(wordtoix[word])

    return x

def make_idx_data(train, test, train_labels, test_labels, wordtoix):
    """
    Transforms sentences into lists.
    """
    tr_x, te_x = [], []
    for rev in train:
        sent = get_idx_from_sent(rev, wordtoix)
        tr_x.append(sent)

    for rev in test:
        sent = get_idx_from_sent(rev, wordtoix)
        te_x.append(sent)


    tr = (tr_x, train_labels)
    te = (te_x, test_labels)
    return tr, te

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, k):
    """
    For words that do not occur in the pretrained word embedding, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pretrained ones
    """
    for word in vocab:
        if word not in word_vecs:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)

def get_W(w2v, ixtoword, k):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(ixtoword)
    W = np.zeros(shape=(vocab_size, k))

    for idx in range(vocab_size):
        W[idx] = w2v[ixtoword[idx]]

    return W
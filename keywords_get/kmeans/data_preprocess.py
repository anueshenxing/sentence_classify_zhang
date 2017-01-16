# coding=utf-8

import cPickle
from collections import defaultdict
import numpy as np
from kmeans_algorithm import KMeansClassifier

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

def add_unknown_words(word_vecs, vocab, k=200):
    """
    For words that do not occur in the pretrained word embedding, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pretrained ones
    """
    for word in vocab:
        if word not in word_vecs:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)

def get_W(w2v, ixtoword, k=200):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(ixtoword)
    W = np.zeros(shape=(vocab_size, k))

    for idx in range(vocab_size):
        W[idx] = w2v[ixtoword[idx]]
    return W

if __name__=="__main__":
    w2v_file = '/home/zhang/PycharmProjects/sentence_classification/news_data/yuliao.bin'
    x = cPickle.load(open("/home/zhang/PycharmProjects/sentence_classification/keywords_get/tf_idf/word_id_duizhaobiao.p", "rb"))
    x_1 = cPickle.load(open("/home/zhang/PycharmProjects/sentence_classification/keywords_get/tf_idf/word_IDF.p", "rb"))
    ixtoword, wordtoix = x[0], x[1]
    all_news_content, vocab_statitic = x_1[0], x_1[1]
    del x, x_1
    w2v = load_bin_vec(w2v_file, wordtoix)
    add_unknown_words(w2v, wordtoix)
    W = get_W(w2v, ixtoword)  # 保存词的标号 与向量的对应关系

    # print W[10000]
    all_news_content_to_wordvec = []
    for i in range(len(all_news_content)):
        news_content_to_wordvec = []
        for j in range(len(all_news_content[i])):
            word_vec = W[all_news_content[i][j]]
            news_content_to_wordvec.append(word_vec)
        all_news_content_to_wordvec.append(news_content_to_wordvec)

    # print all_news_content_to_wordvec[1]
    # cPickle.dump([W, all_news_content_to_wordvec], open("W _and_news_content_wordvec.p", "wb"))

    data_X = np.array(all_news_content_to_wordvec[1]).astype(np.float)
    k = 10
    clf = KMeansClassifier(k)
    clf.fit(data_X)
    clusterAssment = clf._clusterAssment

    # 找出距离各个簇最近的1个词
    result = defaultdict(float)
    min_dist = [10.,10.,10.,10.,10.,10.,10.,10.,10.,10.]
    for i in range(len(clusterAssment)):
        one = clusterAssment[i]
        if one[1] < min_dist[int(one[0])]:
            result[int(one[0])] = i
            min_dist[int(one[0])] = one[1]

    for key in result.keys():
        id = all_news_content[1][result[key]]
        word = ixtoword[id]
        print word
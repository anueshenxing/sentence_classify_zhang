# encoding=utf-8

import cPickle
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

pre_dir = "/home/zhang/PycharmProjects/sentence_classify_zhang/data_file/"
word_flag_vocab_dir = pre_dir + "word_flag_vocab.p"

word_flag_vocab = cPickle.load(open(word_flag_vocab_dir, "rb"))

word_flag = word_flag_vocab[0]
# for key in word_flag:
#     print key + ":" + word_flag[key]

print word_flag[u'报道'.decode("utf-8")]
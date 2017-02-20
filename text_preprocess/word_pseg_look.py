 # encoding=utf-8

import cPickle
import sys
import time
reload(sys)
sys.setdefaultencoding("utf-8")

pre_dir = "/home/zhang/PycharmProjects/sentence_classify_zhang/data_file_2017/"
word_flag_vocab_dir = pre_dir + "word_flag_vocab.p"
wordtoix_and_ixtoword_dir = pre_dir + "wordtoix_and_ixtoword_true.p"
all_news_word_tf_idf_dir = pre_dir + "all_news_word_tf_idf_true.p"
word_vec_dict_dir = pre_dir + "word_vec_dict_true.p"

print time.asctime(time.localtime(time.time()))

word_flag_vocab = cPickle.load(open(word_flag_vocab_dir, "rb"))
wordtoix_and_ixtoword = cPickle.load(open(wordtoix_and_ixtoword_dir, "rb"))
all_news_word_tf_idf = cPickle.load(open(all_news_word_tf_idf_dir, "rb"))
word_vec_dict = cPickle.load(open(word_vec_dict_dir, "rb"))

word_flag = word_flag_vocab[0]
wordtoix, ixtoword = wordtoix_and_ixtoword[0], wordtoix_and_ixtoword[1]
all_news_word_tf_idf = all_news_word_tf_idf[0]
word_vec = word_vec_dict[0]

# del word_flag_vocab, wordtoix_and_ixtoword, all_news_word_tf_idf, word_vec_dict

print time.asctime(time.localtime(time.time()))

# print word_flag[u'全州县'.encode('utf-8')] # 词ci-词ci词性对照表 unicode编码
# print wordtoix[u'全州县'.encode("utf-8")]
# for i in range(10000):
#     print ixtoword[i]

# for key in all_news_word_tf_idf[0].keys():
#     print key + str(all_news_word_tf_idf[0][key])

print word_vec[11111]
print time.asctime(time.localtime(time.time()))


 # encoding=utf-8

import cPickle
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

pre_dir = "/home/zhang/PycharmProjects/sentence_classify_zhang/data_file_2017/"
word_flag_vocab_dir = pre_dir + "word_flag_vocab.p"
all_news_word_tf_idf_and_others_dir = pre_dir + "all_news_word_tf_idf_and_others.p"
word_vec_dict_dir = pre_dir + "word_vec_dict.p"

word_flag_vocab = cPickle.load(open(word_flag_vocab_dir, "rb"))
# all_news_word_tf_idf_and_others = cPickle.load(open(all_news_word_tf_idf_and_others_dir, "rb"))
# word_vec_dict = cPickle.load(open(word_vec_dict_dir, "rb"))

word_flag = word_flag_vocab[0]
# wordtoix, ixtoword, all_news_word_tf_idf = all_news_word_tf_idf_and_others[0], \
#                                            all_news_word_tf_idf_and_others[1], all_news_word_tf_idf_and_others[4]

# word_vec = word_vec_dict[0]
# for key in word_flag:
#     print key + ":" + word_flag[key]

print word_flag[u'全州县'] # 词ci-词ci词词词词词词性对照表 unicode编码
# print wordtoix[u'全州县'.encode("utf-8")]
# for i in range(10000):
#     print ixtoword[i]

# for key in all_news_word_tf_idf[0].keys():
#     print key + str(all_news_word_tf_idf[0][key])

# print word_vec[111]


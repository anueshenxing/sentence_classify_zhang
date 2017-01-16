# encoding=utf-8

import cPickle
import jieba
import jieba.posseg as pseg
from pymongo import *
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
pre_dir_1 = "/home/zhang/PycharmProjects/sentence_classification/data_file/"
pre_dir = "/home/zhang/PycharmProjects/sentence_classification/data_file/20170111/"
all_news_word_tf_idf_and_others_dir = pre_dir + "all_news_word_tf_idf_and_others_2017011.p"
word_vec_dict_dir = pre_dir_1 + "word_vec_dict.p"

all_news_word_tf_idf_and_others = cPickle.load(open(all_news_word_tf_idf_and_others_dir, "rb"))
word_vec_dict = cPickle.load(open(word_vec_dict_dir, "rb"))

wordtoix, ixtoword, all_news_content_by_id, all_news_word_tf_idf = \
        all_news_word_tf_idf_and_others[0], all_news_word_tf_idf_and_others[1], \
        all_news_word_tf_idf_and_others[3], all_news_word_tf_idf_and_others[4]
for i in range(len(all_news_word_tf_idf[0])):
    print all_news_word_tf_idf[1][i][0] + ":" + str(all_news_word_tf_idf[1][i][1])
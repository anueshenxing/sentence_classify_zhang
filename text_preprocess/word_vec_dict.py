# coding=utf-8
"""
    生成 词-词向量对照表：word_vec_dict,保存词的标号 与向量的对应关系
"""
import cPickle
from collections import defaultdict

import time

from util.tools import *

reload(sys)
sys.setdefaultencoding("utf-8")

if __name__=="__main__":
    pre_dir = "/home/zhang/PycharmProjects/sentence_classify_zhang/data_file_2017/"
    wordtoix_and_ixtoword_dir = pre_dir + "wordtoix_and_ixtoword.p"
    w2v_file_dir = pre_dir + "yuliao_20170208_100.bin"

    print time.asctime(time.localtime(time.time()))
    wordtoix_and_ixtoword = cPickle.load(open(wordtoix_and_ixtoword_dir, "rb"))
    wordtoix, ixtoword = wordtoix_and_ixtoword[0], wordtoix_and_ixtoword[1]
    print time.asctime(time.localtime(time.time()))

    w2v = load_bin_vec(w2v_file_dir, wordtoix)
    add_unknown_words(w2v, wordtoix, 100)
    W = get_W(w2v, ixtoword, 100)  # 保存词的标号 与向量的对应关系

    print time.asctime(time.localtime(time.time()))

    cPickle.dump([W], open(pre_dir + "word_vec_dict_true.p", "wb"), True)

    print time.asctime(time.localtime(time.time()))


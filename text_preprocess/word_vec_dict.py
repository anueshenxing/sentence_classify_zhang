# coding=utf-8
"""
    生成 词-词向量对照表：word_vec_dict,保存词的标号 与向量的对应关系
"""
import cPickle
from collections import defaultdict
from util.tools import *

reload(sys)
sys.setdefaultencoding("utf-8")

if __name__=="__main__":
    pre_dir = "/home/zhang/PycharmProjects/sentence_classify_zhang/data_file/"
    all_news_word_tf_idf_and_others_dir = pre_dir + "all_news_word_tf_idf_and_others.p"
    w2v_file_dir = pre_dir + "yuliao_20170111_100.bin"

    all_news_word_tf_idf_and_others = cPickle.load(open(all_news_word_tf_idf_and_others_dir, "rb"))
    wordtoix, ixtoword = all_news_word_tf_idf_and_others[0], all_news_word_tf_idf_and_others[1]
    w2v = load_bin_vec(w2v_file_dir, wordtoix)
    add_unknown_words(w2v, wordtoix, 100)
    W = get_W(w2v, ixtoword, 100)  # 保存词的标号 与向量的对应关系
    cPickle.dump([W], open(pre_dir + "word_vec_dict.p", "wb"))

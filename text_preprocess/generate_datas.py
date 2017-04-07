# encoding=utf-8
"""
    功能：生成关键词+新闻标题数据集，分别采用TF-IDF，TF-CRF，语义复杂网络，多特征融合方法
"""
import random
import cPickle
import time
import numpy as np
from feature_abstract_algorithm_contrast.WeightCompute import WeightCompute
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

def tf_idf_data():
    computeW = WeightCompute()
    pre_dir = "/home/zhang/PycharmProjects/sentence_classify_zhang/data_file_2017/"
    print "程序开始时间：" + time.asctime(time.localtime(time.time()))

    # 加载文件地址
    wordtoix_and_ixtoword_dir = pre_dir + "wordtoix_and_ixtoword_true.p"
    all_news_content_by_id_dir = pre_dir + "all_news_content_by_id_true.p"
    word_vec_dict_dir = pre_dir + "word_vec_dict_true.p"
    news_title_category_dir = pre_dir + "news_title_category.p"
    tf_idf_dir = pre_dir + "all_news_word_tf_idf_true.p"
    word_flag_vocab_dir = pre_dir + "word_flag_vocab.p"  # 词性词典
    word_pseg_dir = pre_dir + "word_flag_lsit.txt"  # 有效词性列表

    # 加载文件
    wordtoix_and_ixtoword_p = cPickle.load(open(wordtoix_and_ixtoword_dir, "rb"))
    all_news_content_by_id_p = cPickle.load(open(all_news_content_by_id_dir, "rb"))
    word_vec_dict_p = cPickle.load(open(word_vec_dict_dir, "rb"))
    news_title_category_p = cPickle.load(open(news_title_category_dir, "rb"))
    tf_idf_p = cPickle.load(open(tf_idf_dir, "rb"))
    word_flag_vocab_p = cPickle.load(open(word_flag_vocab_dir, "rb"))

    # 获取需要的数据
    wordtoix, ixtoword = wordtoix_and_ixtoword_p[0], wordtoix_and_ixtoword_p[1]
    all_news_content_by_id = all_news_content_by_id_p[0]
    W = word_vec_dict_p[0]  # 保存词的标号 与向量的对应关系
    news_title_category = news_title_category_p[0]  # 新闻标题与类别
    tf_idf = tf_idf_p[0]  # 保存已经计算好的tf-idf值
    word_flag = word_flag_vocab_p[0]  # 保存词语编号与词性的对应关系
    word_pseg_list = computeW.load_useful_word_psegs(word_pseg_dir)

    del wordtoix_and_ixtoword_p, all_news_content_by_id_p, word_vec_dict_p, news_title_category_p, tf_idf_p
    print "程序加载数据完毕时间：" + time.asctime(time.localtime(time.time()))

    print news_title_category[0]

if __name__ == "__main__":
    tf_idf_data()

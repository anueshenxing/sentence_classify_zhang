# coding=utf-8
import cPickle
import sys

import time

from collections import defaultdict

from WeightCompute import WeightCompute

reload(sys)
sys.setdefaultencoding("utf-8")

if __name__ == "__main__":
    """
        基于多特征融合的关键词提取
    """
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
    news_select = {'society': [0, 1, 18, 19, 23, 29, 37, 46, 70, 92, 137, 308],
                   'edu': [1488, 1663, 1745, 1763, 1838, 1851, 1857, 1900],
                   'sports': [3387, 3438, 3776, 3779, 3874, 3851, 3856, 3915, 4192],
                   'travel': [20061, 20215, 20425, 20457, 20578, 20607, 20783, 20833],
                   'military': [53470, 53494, 53516, 53523, 53738, 53942, 54139, 54386],
                   'finance': [50035, 50044, 51574, 51722, 52159, 52641, 53057, 53671, 54323],
                   'tech': [40016, 35049, 35056, 35324, 35346, 35604, 35796, 35852, 35915],
                   'food': [35316, 35322, 35390, 35819, 36121, 36786, 37257, 38839],
                   'health': [55400, 55401, 56251, 56622, 57341, 58680, 59137, 59516, 59893],
                   'car': [70307, 62874, 63618, 63849, 64424, 64510, 64588],
                   'entertainment': [64687, 63998, 61329, 61948, 63494, 63500, 63713, 63902]}

    result_dir = "/home/zhang/Desktop/keywords.txt"
    result_file = open(result_dir, 'a')
    for key in news_select.keys():
        category_text = key + ":" + "\n"
        result_file.write(category_text)
        for news_index in news_select[key]:
            news_tf_idf = computeW.compute_tfidf(news_index, tf_idf)
            news_crf = computeW.compute_crf(news_index, news_title_category, ixtoword, all_news_content_by_id)
            news_cn = computeW.compute_complete_network(news_index, all_news_content_by_id, W, ixtoword)

            news_w = defaultdict(float)
            for key in news_tf_idf.keys():
                # key 为文字
                news_title_word_list = computeW.get_news_title_word_list(news_index, news_title_category)
                news_title_word_by_id_list_cleaned = computeW.clean_title(news_title_word_list, word_flag,
                                                                          word_pseg_list, wordtoix)
                weight_rt = computeW.function_relating(news_title_word_by_id_list_cleaned, W, wordtoix[key])

                weight_pseg = computeW.function_pseg(word_pseg_list, word_flag[key])
                # print news_tf_idf[key]
                # print news_crf[key]
                # print news_cn[key]
                # print weight_rt
                # print weight_pseg
                # print '=========================================='
                weight = weight_pseg * weight_rt * news_crf[key] * news_cn[key]
                news_w[key] = weight
            print "程序执行完毕时间：" + time.asctime(time.localtime(time.time()))
            news_w = sorted(news_w.iteritems(), key=lambda d: d[1], reverse=True)
            keywords = []
            count = 0
            for i in range(len(news_w)):
                # print tf_idf_of_index[i][0] + ":" + str(tf_idf_of_index[i][1])
                keywords.append(news_w[i][0])
                count += 1
                if count > 20:
                    break
            keywords_text = "news_index:" + str(news_index) + ":" + " ".join(keywords) + "\n"
            result_file.write(keywords_text)
            # for i in range(len(news_w)):
                # print news_w[i][0] + ":" + str(news_w[i][1])



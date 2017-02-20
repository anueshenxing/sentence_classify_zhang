# coding=utf-8
"""
    计算TF-IDF权值，并排序，保存结果至TF_IDF.txt文件中
"""
import cPickle
import time
from util.tools import *

reload(sys)
sys.setdefaultencoding("utf-8")

if __name__ == "__main__":
    pre_dir = "/home/zhang/PycharmProjects/sentence_classify_zhang/data_file_2017/"
    result_pre_dir = "/home/zhang/PycharmProjects/sentence_classify_zhang/feature_abstract_result/"

    tf_idf_result_dir = result_pre_dir + "TF_IDF.txt"
    tf_idf_dir = pre_dir + "all_news_word_tf_idf_true.p"
    news_title_category_dir = pre_dir + "news_title_category.p"

    news_title_category_p = cPickle.load(open(news_title_category_dir, "rb"))
    tf_idf_p = cPickle.load(open(tf_idf_dir, "rb"))
    tf_idf = tf_idf_p[0]
    news_title_category = news_title_category_p[0]
    del tf_idf_p, news_title_category_p
    TF_IDF_result = open(tf_idf_result_dir, "a")

    for index in range(len(tf_idf)):  # len(tf_idf)
        if index % 1000 == 0:
            print "index -> " + str(index)
        category = news_title_category[index].split()[0]

        tf_idf_of_index = tf_idf[index]
        tf_idf_of_index = sorted(tf_idf_of_index.iteritems(), key=lambda d: d[1], reverse=True)
        features = []
        count = 0
        for i in range(len(tf_idf_of_index)):
            # print tf_idf_of_index[i][0] + ":" + str(tf_idf_of_index[i][1])
            features.append(tf_idf_of_index[i][0])
            count += 1
            if count > 20:
                break
        result_of_index = "  ".join(features) + "\n"
        result_of_index = category + " " + result_of_index
        result_of_index.encode("utf-8")
        TF_IDF_result.write(result_of_index)
    TF_IDF_result.close()

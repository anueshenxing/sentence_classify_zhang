# coding=utf-8
"""
    计算CRF权值，并排序，保存结果至CRF.txt文件中
    由于计算一条新闻需要2min左右比较慢，因此选取其中的一部分进行验证
"""
import cPickle
import time
from util.tools import *

reload(sys)
sys.setdefaultencoding("utf-8")


def count(all_news_content_by_id,news_title_category,ID_of_word,category):
    """
    :param all_news_content_by_id:
    :param news_title_category:
    :param ID_of_word:
    :param category:
    :return: 包含词ID_of_word，且类别为category的新闻数量
    """
    count_X = 0
    count_P = 0
    for index in range(len(all_news_content_by_id)):
        one_news_by_id = all_news_content_by_id[index]
        id_category = news_title_category[index].split()[0].encode("utf-8")
        category = category.encode("utf-8")
        if ID_of_word in one_news_by_id and category == id_category:
            count_X += 1
        if ID_of_word in one_news_by_id and category != id_category:
            count_P += 1

    return count_X, count_P


if __name__ == "__main__":
    pre_dir = "/home/zhang/PycharmProjects/sentence_classify_zhang/data_file_2017/"
    result_pre_dir = "/home/zhang/PycharmProjects/sentence_classify_zhang/feature_abstract_result/"

    crf_result_dir = result_pre_dir + "CRF.txt"
    CRF_result = open(crf_result_dir, "a")

    news_title_category_dir = pre_dir + "news_title_category.p"
    news_title_category_p = cPickle.load(open(news_title_category_dir, "rb"))
    news_title_category = news_title_category_p[0]

    wordtoix_and_ixtoword_dir = pre_dir + "wordtoix_and_ixtoword_true.p"
    wordtoix_and_ixtoword_p = cPickle.load(open(wordtoix_and_ixtoword_dir, "rb"))
    ixtoword = wordtoix_and_ixtoword_p[1]

    all_news_content_by_id_dir = pre_dir + "all_news_content_by_id_true.p"
    all_news_content_by_id_p = cPickle.load(open(all_news_content_by_id_dir, "rb"))
    all_news_content_by_id = all_news_content_by_id_p[0]

    del news_title_category_p, all_news_content_by_id_p, wordtoix_and_ixtoword_p

    b = 0.00001
    Sum_all_news = 84958  # 新闻总数
    sum_category = {"society": 8764, "edu": 9223, "sports": 6991, "travel": 7887,
                    "military": 5180, "finance": 9470, "tech": 8959, "food": 7554,
                    "health": 6807, "car": 7409}
    print time.asctime(time.localtime(time.time()))

    for index in range(1):  # len(all_news_content_by_id)
        n = len(all_news_content_by_id[index])
        news_word_set = set(all_news_content_by_id[index])
        category = news_title_category[index].split()[0]
        word_crf = defaultdict(float)
        for word_id in news_word_set:
            tf = all_news_content_by_id[index].count(word_id) / float(n)
            X, P = count(all_news_content_by_id, news_title_category, word_id, category)
            Y = sum_category[category]
            Q = Sum_all_news - Y
            X = float(X)
            CRF = math.log((X/Y)/((P/Q) + b))
            tf_CRF = tf * CRF
            word_crf[ixtoword[word_id]] = tf_CRF

        word_crf = sorted(word_crf.iteritems(), key=lambda d: d[1], reverse=True)
        features = []
        count = 0
        for i in range(len(word_crf)):
            # print word_crf[i][0] + ":" + str(word_crf[i][1])
            features.append(word_crf[i][0])
            count += 1
            if count > 20:
                break
        result_of_index = "  ".join(features) + "\n"
        result_of_index = category + " " + result_of_index
        result_of_index.encode("utf-8")
        CRF_result.write(result_of_index)
    CRF_result.close()
    print time.asctime(time.localtime(time.time()))






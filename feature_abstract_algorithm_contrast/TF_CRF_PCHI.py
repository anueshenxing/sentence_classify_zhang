# coding=utf-8
"""
    计算TF-CRF-PCHI权值，并排序，保存结果至TF-CRF-PCHI.txt文件中
    由于计算一条新闻需要2min左右比较慢，因此选取其中的一部分进行验证
    注意：是结果不好
"""
import cPickle
import time
from util.tools import *

reload(sys)
sys.setdefaultencoding("utf-8")


def count(all_news_content_by_id, news_title_category, ID_of_word, category):
    """
    :param all_news_content_by_id:
    :param news_title_category:
    :param ID_of_word:
    :param category:
    :return: 包含词ID_of_word，且类别为category的新闻数量
    """
    Sum_all_news = 84958  # 新闻总数
    count_D11 = 0  # 包含词语t，类别为C的文本的数量
    count_D12 = 0  # 包含词语t，类别不是C的文本的数量
    count_D21 = 0  # 不包含词语t，类别为C的文本的数量
    count_D22 = 0  # 不包含词语t，类别不是C的文本的数量
    count_txt_with_word = 0  # 包含词语t的文本数量

    for index in range(len(all_news_content_by_id)):
        one_news_by_id = all_news_content_by_id[index]
        id_category = news_title_category[index].split()[0].encode("utf-8")
        category = category.encode("utf-8")
        if ID_of_word in one_news_by_id and category == id_category:
            count_D11 += 1
        if ID_of_word in one_news_by_id and category != id_category:
            count_D12 += 1
        if ID_of_word not in one_news_by_id and category == id_category:
            count_D21 += 1
        if ID_of_word in one_news_by_id:
            count_txt_with_word += 1
    count_D22 = Sum_all_news - count_D11 - count_D12 - count_D21

    return float(count_D11), float(count_D12), float(count_D21), float(count_D22), float(count_txt_with_word)


if __name__ == "__main__":
    pre_dir = "/home/zhang/PycharmProjects/sentence_classify_zhang/data_file_2017/"
    result_pre_dir = "/home/zhang/PycharmProjects/sentence_classify_zhang/feature_abstract_result/"

    crf_result_dir = result_pre_dir + "TCPC.txt"
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

    for index in range(1,2):  # len(all_news_content_by_id)
        n = len(all_news_content_by_id[index])
        news_word_set = set(all_news_content_by_id[index])
        category = news_title_category[index].split()[0]
        word_tf_crf_pchi = defaultdict(float)
        for word_id in news_word_set:
            log_tf = math.log(all_news_content_by_id[index].count(word_id) / float(n))
            D11, D12, D21, D22, count_txt_with_word = count(all_news_content_by_id, news_title_category, word_id,
                                                            category)
            T11 = (D11 + D12) * ((D11 + D21) / Sum_all_news)
            T12 = (D11 + D12) * ((D12 + D22) / Sum_all_news)
            T21 = (D21 + D22) * ((D11 + D21) / Sum_all_news)
            T22 = (D21 + D22) * ((D12 + D22) / Sum_all_news)
            Y = sum_category[category]
            Q = Sum_all_news - Y

            CRF = math.log((D11 / Y) / ((D12 / Q) + b))

            CHI = (math.pow((D11 - T11), 2) / T11) + (math.pow((D12 - T12), 2) / T12) + (
                math.pow((D21 - T21), 2) / T21) + (math.pow((D22 - T22), 2) / T22)

            u = D11 - (count_txt_with_word / 11)
            if u > 0:
                u = 1.
            else:
                u = -1.

            PCHI = u * CHI
            tf_crf_pchi = log_tf * CRF * PCHI
            word_tf_crf_pchi[ixtoword[word_id]] = tf_crf_pchi

        word_tf_crf_pchi = sorted(word_tf_crf_pchi.iteritems(), key=lambda d: d[1], reverse=True)
        for i in range(len(word_tf_crf_pchi)):
            print word_tf_crf_pchi[i][0] + ":" + str(word_tf_crf_pchi[i][1])

    print time.asctime(time.localtime(time.time()))

# coding=utf-8
"""
    根据新闻内容中词的语义关系构建复杂网络，G(V, E),其中V是新闻内容中词的集合，E是V中两词之间的边（两词的语义相似度大于b时，建立连接）
"""
import cPickle
from collections import defaultdict
import numpy as np
import time

from util.tools import *
import json

reload(sys)
sys.setdefaultencoding("utf-8")


def CosineDist(arrA, arrB):
    """
        功能：cosine距离距离计算
        输入：两个一维数组
        """
    return np.dot(arrA, arrB) / (np.linalg.norm(arrA) * np.linalg.norm(arrB))


def load_useful_word_psegs(file_dir):
    """
        加载有用的词性列表
    """
    word_pseg_file = open(file_dir, 'rb')
    word_pseg_list = []
    for line in word_pseg_file:
        word_pseg = line.split()[0]
        word_pseg_list.append(word_pseg)
    return word_pseg_list


def load_all_news_title_by_id(file_dir, wordtoix):
    """
        加载新闻标题，并将新闻标题以词id来表示
    """
    all_news_title_and_ctg = open(all_news_title_and_ctg_dir, 'rb')
    all_news_title_by_id = []
    for sent in all_news_title_and_ctg:
        news_title_by_id = []
        sent = sent.split()[1:]
        for word in sent:
            news_title_by_id.append(wordtoix[word])
        all_news_title_by_id.append(news_title_by_id)
    return all_news_title_by_id


def clean_title(all_news_title_by_id, word_flag, word_pseg_list, ixtoword):
    """
        去除标题中没有意义的词
    """
    all_news_title_by_id_cleaned = []

    for title in all_news_title_by_id:
        news_title_by_id_cleaned = []
        for word in title:
            pseg = word_flag[ixtoword[word]]
            if pseg in word_pseg_list:
                news_title_by_id_cleaned.append(word)
        all_news_title_by_id_cleaned.append(news_title_by_id_cleaned)
    return all_news_title_by_id_cleaned


def function_pseg(word_pseg_list, word_pseg):
    """
        词性系数
    """
    f = 0.
    if word_pseg in word_pseg_list:
        f = 1.
    return f


def function_relating(V_title, W, word_id):
    """
        计算新闻内容中的词 与 标题中所有词 的 最大语义相关度
    """
    max_relating = 0.1
    for id in V_title:
        arrA = W[word_id]
        arrB = W[id]
        arrA = np.asarray(W[Vi])
        arrB = np.asarray(W[Vj])
        cosine_dist = CosineDist(arrA, arrB)
        if cosine_dist > max_relating:
            max_relating = cosine_dist
    return max_relating


def function_compute_word_weight(wordtoix, ixtoword, all_news_word_tf_idf,
                                 all_news_content_by_id, word_flag, W,
                                 word_pseg_list, all_news_title_by_id_cleaned):
    """
        计算新闻内容中词的综合权重
    :return: 新闻内容中词的综合权重排序列表
    """

    return ""


# if __name__ == "__main__":
#     pre_dir = "/home/zhang/PycharmProjects/sentence_classify_zhang/data_file/"
#     word_pseg_dir = pre_dir + "word_flag_lsit.txt"
#     word_pseg_list = load_useful_word_psegs(word_pseg_dir)
#     print function_pseg(word_pseg_list, "n")

if __name__ == "__main__":
    pre_dir = "/home/zhang/PycharmProjects/sentence_classify_zhang/data_file_2017/"

    print "程序开始时间：" + time.asctime(time.localtime(time.time()))

    # 加载文件地址
    wordtoix_and_ixtoword_dir = pre_dir + "wordtoix_and_ixtoword_true.p"
    all_news_word_tf_idf_dir = pre_dir + "all_news_word_tf_idf_true.p"
    all_news_content_by_id_dir = pre_dir + "all_news_content_by_id_true.p"
    word_vec_dict_dir = pre_dir + "word_vec_dict_true.p"
    word_pseg_dir = pre_dir + "word_flag_lsit.txt"
    word_flag_vocab_dir = pre_dir + "word_flag_vocab.p"
    all_news_title_and_ctg_dir = pre_dir + "news_title_data_and_category.txt"
    news_title_category_dir = pre_dir + "news_title_category.p"
    all_news_title_and_ctg_with_keywords_dir = pre_dir + "all_news_title_and_ctg_with_keywords.txt"

    # 加载文件
    wordtoix_and_ixtoword_p = cPickle.load(open(wordtoix_and_ixtoword_dir, "rb"))
    all_news_word_tf_idf_p = cPickle.load(open(all_news_word_tf_idf_dir, "rb"))
    all_news_content_by_id_p = cPickle.load(open(all_news_content_by_id_dir, "rb"))
    word_vec_dict_p = cPickle.load(open(word_vec_dict_dir, "rb"))
    word_flag_vocab_p = cPickle.load(open(word_flag_vocab_dir, "rb"))
    news_title_category_p = cPickle.load(open(news_title_category_dir, "rb"))
    all_news_title_and_ctg_with_keywords = open(all_news_title_and_ctg_with_keywords_dir, "a")

    # 获取需要的数据
    wordtoix, ixtoword = wordtoix_and_ixtoword_p[0], wordtoix_and_ixtoword_p[1]
    all_news_word_tf_idf = all_news_word_tf_idf_p[0]
    all_news_content_by_id = all_news_content_by_id_p[0]
    word_flag = word_flag_vocab_p[0]
    W = word_vec_dict_p[0]  # 保存词的标号 与向量的对应关系
    news_title_category = news_title_category_p[0]
    word_pseg_list = load_useful_word_psegs(word_pseg_dir)
    all_news_title_by_id = load_all_news_title_by_id(all_news_title_and_ctg_dir, wordtoix)
    all_news_title_by_id_cleaned = clean_title(all_news_title_by_id, word_flag, word_pseg_list, ixtoword)

    del wordtoix_and_ixtoword_p, all_news_word_tf_idf_p, all_news_content_by_id_p, word_vec_dict_p, word_flag_vocab_p
    print "程序加载数据完毕时间：" + time.asctime(time.localtime(time.time()))

    num_of_all_news = len(all_news_content_by_id)
    for index_of_news in range(61502, num_of_all_news):
        if index_of_news % 100 == 0:
            print index_of_news

        V = all_news_content_by_id[index_of_news]

        # 新闻内容中词的个数
        n = len(V)

        # 两词的语义相似度大于b时，建立连接
        b = 0.5

        # 新闻内容中的词的连接矩阵
        E = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i, n):
                Vi = V[i]  # 词的id
                Vj = V[j]  # 词的id
                arrA = np.asarray(W[Vi])
                arrB = np.asarray(W[Vj])
                cosine_dist = CosineDist(arrA, arrB)
                # print str(cosine_dist) + ":" +ixtoword[Vi] + " " + ixtoword[Vj]Vj
                if i == j:
                    E[i][j] = 1.
                if cosine_dist > b:
                    E[i][j] = cosine_dist
                    E[j][i] = cosine_dist

        # 计算复杂网络节点重要性指标
        IMP = defaultdict(float)  # 保存新闻内容中词的综合权重
        Vi_NeiContribution = []  # 新闻内容中每个词Vi的所有邻居节点贡献率之和
        sumAll_NeiContribution = 0.  # 新闻内容中所有词的邻居节点贡献率之和
        for i in range(n):
            NeiContribution = 0.
            for j in range(n):
                if i != j and E[i][j] > 0.5:
                    freq_j = float(all_news_word_tf_idf[index_of_news][ixtoword[V[j]]])
                    strength_i_j = float(E[i][j])
                    authority = freq_j * strength_i_j
                    NeiContribution += authority
                else:
                    NeiContribution += 0.
            Vi_NeiContribution.append(NeiContribution)
            sumAll_NeiContribution += NeiContribution

        # 得到新闻内容中每个词基于复杂网络的权重
        for i in range(n):
            if sumAll_NeiContribution == 0.:
                sumAll_NeiContribution = 1.
            IMP[ixtoword[V[i]]] = Vi_NeiContribution[i] / sumAll_NeiContribution

        # 添加词性系数
        for i in range(n):
            pseg = word_flag[ixtoword[V[i]].encode("utf-8")]
            IMP[ixtoword[V[i]]] *= function_pseg(word_pseg_list, pseg)

        # 添加与标题相关度系数
        V_title = all_news_title_by_id_cleaned[index_of_news]  # 对应的新闻标题
        for i in range(n):
            IMP[ixtoword[V[i]]] *= function_relating(V_title, W, V[i])

        # 添加tf_idf系数
        for i in range(n):
            IMP[ixtoword[V[i]]] *= all_news_word_tf_idf[index_of_news][ixtoword[V[i]]]

        # 将提取到的关键词直接添加到新闻标题结尾
        one_news_title_with_keywords = news_title_category[index_of_news].split('\n')[0]

        IMP = sorted(IMP.iteritems(), key=lambda d: d[1], reverse=True)
        len_of_IMP = len(IMP)
        if len_of_IMP == 0:
            one_news_title_with_keywords += '\n'
        elif 0 < len_of_IMP < 15:
            keywords = []
            for i in range(len(IMP)):
                keywords.append(IMP[i][0].encode('utf-8'))
            keywords_txt = " ".join(keywords) + "\n"
            one_news_title_with_keywords = one_news_title_with_keywords + " " + keywords_txt
        else:
            keywords = []
            for i in range(15):
                keywords.append(IMP[i][0].encode('utf-8'))
            keywords_txt = " ".join(keywords) + "\n"
            one_news_title_with_keywords = one_news_title_with_keywords + " " + keywords_txt

        # for i in range(len(IMP)):
        #     print IMP[i][0] + ":" + str(IMP[i][1])
        print one_news_title_with_keywords
        all_news_title_and_ctg_with_keywords.write(one_news_title_with_keywords)

    print "程序执行完毕时间：" + time.asctime(time.localtime(time.time()))


# if __name__ == "__main__":
#     pre_dir = "/home/zhang/PycharmProjects/sentence_classify_zhang/data_file_2017/"
#
#     print "程序开始时间：" + time.asctime(time.localtime(time.time()))
#
#     # 加载文件地址
#     wordtoix_and_ixtoword_dir = pre_dir + "wordtoix_and_ixtoword_true.p"
#     all_news_word_tf_idf_dir = pre_dir + "all_news_word_tf_idf_true.p"
#     all_news_content_by_id_dir = pre_dir + "all_news_content_by_id_true.p"
#     word_vec_dict_dir = pre_dir + "word_vec_dict_true.p"
#     word_pseg_dir = pre_dir + "word_flag_lsit.txt"
#     word_flag_vocab_dir = pre_dir + "word_flag_vocab.p"
#     all_news_title_and_ctg_dir = pre_dir + "news_title_data_and_category.txt"
#
#     # 加载文件
#     wordtoix_and_ixtoword_p = cPickle.load(open(wordtoix_and_ixtoword_dir, "rb"))
#     all_news_word_tf_idf_p = cPickle.load(open(all_news_word_tf_idf_dir, "rb"))
#     all_news_content_by_id_p = cPickle.load(open(all_news_content_by_id_dir, "rb"))
#     word_vec_dict_p = cPickle.load(open(word_vec_dict_dir, "rb"))
#     word_flag_vocab_p = cPickle.load(open(word_flag_vocab_dir, "rb"))
#
#     # 获取需要的数据
#     wordtoix, ixtoword = wordtoix_and_ixtoword_p[0], wordtoix_and_ixtoword_p[1]
#     all_news_word_tf_idf = all_news_word_tf_idf_p[0]
#     all_news_content_by_id = all_news_content_by_id_p[0]
#     word_flag = word_flag_vocab_p[0]
#     W = word_vec_dict_p[0]  # 保存词的标号 与向量的对应关系
#     word_pseg_list = load_useful_word_psegs(word_pseg_dir)
#     all_news_title_by_id = load_all_news_title_by_id(all_news_title_and_ctg_dir, wordtoix)
#     all_news_title_by_id_cleaned = clean_title(all_news_title_by_id, word_flag, word_pseg_list, ixtoword)
#
#     print "程序加载数据完毕时间：" + time.asctime(time.localtime(time.time()))
#
#     index_of_news = 0
#
#
#
#     V = all_news_content_by_id[index_of_news]
#
#     # 新闻内容中词的个数
#     n = len(V)
#
#     # 两词的语义相似度大于b时，建立连接
#     b = 0.5
#
#     # 新闻内容中的词的连接矩阵
#     E = np.zeros((n, n), dtype=np.float32)
#
#     for i in range(n):
#         for j in range(i, n):
#             Vi = V[i]  # 词的id
#             Vj = V[j]  # 词的id
#             arrA = np.asarray(W[Vi])
#             arrB = np.asarray(W[Vj])
#             cosine_dist = CosineDist(arrA, arrB)
#             # print str(cosine_dist) + ":" +ixtoword[Vi] + " " + ixtoword[Vj]Vj
#             if i == j:
#                 E[i][j] = 1.
#             if cosine_dist > b:
#                 E[i][j] = cosine_dist
#                 E[j][i] = cosine_dist
#
#     """
#         生成复杂网络数据版本2
#     """
#     webkit_dep = defaultdict()
#     nodes = []
#     links = []
#
#     for i in range(n):
#         node_dict = defaultdict()
#         node_dict["category"] = 0
#         node_dict["name"] = ixtoword[V[i]].encode("utf-8")
#         nodes.append(node_dict)
#         for j in range(i + 1, n):
#             similarity = E[i][j]
#             if similarity > 0.5:
#                 relate_dict = defaultdict()
#                 relate_dict["source"] = i
#                 relate_dict["target"] = j
#                 relate_dict["weight"] = 1
#                 links.append(relate_dict)
#
#     webkit_dep["nodes"] = nodes
#     webkit_dep["links"] = links
#     with open(pre_dir + 'complete_network.json', 'wb') as f:
#         json.dump(webkit_dep, f)

# coding=utf-8
import sys
import math
import numpy as np
from collections import defaultdict

reload(sys)
sys.setdefaultencoding("utf-8")


class WeightCompute(object):
    def __init__(self):
        self.b = 0.00001
        self.similarity_stand = 0.5  # 两词的语义相似度大于b时，建立连接
        self.sum_all_news = 84958
        self.sum_category = {"society": 8764, "edu": 9223, "sports": 6991, "travel": 7887,
                             "military": 5180, "finance": 9470, "tech": 8959, "food": 7554,
                             "health": 6807, "car": 7409, "entertainment": 6714}

    def compute_tfidf(self, news_index, tf_idf):
        """

        :param news_index: 新闻编号
        :param tf_idf: 已经计算好的tf-idf结果
        :return: 编号为news_index的新闻内容中词语的tf-idf值
        """
        tf_idf_of_index = tf_idf[news_index]
        return tf_idf_of_index

    def compute_crf(self, news_index, news_title_category, ixtoword, all_news_content_by_id):
        """
        :param news_index: 新闻编号
        :param news_title_category: 新闻标题与新闻类别，由具体词语表示
        :param ixtoword: 标号->词语
        :param all_news_content_by_id: 用词语标号表示的所有新闻内容
        :return: 编号为news_index的新闻内容中词语的CRF值
        """
        b = 0.00001
        sum_all_news = 84958  # 新闻总数
        sum_category = {"society": 8764, "edu": 9223, "sports": 6991, "travel": 7887,
                        "military": 5180, "finance": 9470, "tech": 8959, "food": 7554,
                        "health": 6807, "car": 7409}
        n = len(all_news_content_by_id[news_index])
        news_word_set = set(all_news_content_by_id[news_index])
        category = news_title_category[news_index].split()[0]
        word_crf = defaultdict(float)
        for word_id in news_word_set:
            tf = all_news_content_by_id[news_index].count(word_id) / float(n)
            X, P = WeightCompute.count(self, all_news_content_by_id, news_title_category, word_id, category)
            Y = self.sum_category[category]
            Q = self.sum_all_news - Y
            X = float(X)
            CRF = math.log((X / Y) / ((P / Q) + self.b))
            tf_CRF = tf * CRF
            word_crf[ixtoword[word_id]] = tf_CRF
        return word_crf

    def compute_complete_network(self, news_index, all_news_content_by_id, W, ixtoword):
        """
        :param news_index: 新闻编号
        :param all_news_content_by_id: 用词语标号表示的所有新闻内容
        :param W: 词语标号-词向量字典
        :return: 编号为news_index的新闻内容中词语的基于语义复杂网络的权值
        """
        one_news_by_id = all_news_content_by_id[news_index]
        one_news_by_id_word_set = set(one_news_by_id)
        one_news_by_id_word_set_to_list = list(one_news_by_id_word_set)
        # 新闻内容中词的个数
        n = len(one_news_by_id_word_set)

        # 新闻内容中的词的连接矩阵
        E = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i, n):
                word_i = one_news_by_id_word_set_to_list[i]
                word_j = one_news_by_id_word_set_to_list[j]
                arrA = np.asarray(W[word_i])
                arrB = np.asarray(W[word_j])
                cosine_dist = WeightCompute.CosineDist(self, arrA, arrB)
                if i == j:
                    E[i][j] = 1.
                if cosine_dist > self.similarity_stand:
                    E[i][j] = cosine_dist
                    E[j][i] = cosine_dist

        # 计算复杂网络节点重要性指标
        Weight_of_CN = defaultdict(float)  # 保存新闻内容中词的综合权重
        for i in range(n):
            neighbor = []  # 邻居节点集合
            freq_i = float(one_news_by_id.count(one_news_by_id_word_set_to_list[i])) / len(one_news_by_id)
            NeiContribution = 0.  # 加权度的计算
            for j in range(n):
                if i != j and E[i][j] > 0.5:
                    neighbor.append(j)
                    freq_j = float(one_news_by_id.count(one_news_by_id_word_set_to_list[j])) / len(one_news_by_id)
                    strength_i_j = float(E[i][j])
                    authority = freq_j * strength_i_j
                    NeiContribution += authority
                else:
                    NeiContribution += 0.

            # # 计算节点的聚合系数
            # cluster_coe = 0.  # 聚类系数
            # num_of_neighbor = len(neighbor)
            # # 计算邻居节点间的连接数
            # for p in range(num_of_neighbor):
            #     for q in range(p + 1, num_of_neighbor):
            #         if E[p][q] > 0.5:
            #             cluster_coe += 1
            # if len(neighbor) > 1:
            #     cluster_coe = 2 * cluster_coe / (num_of_neighbor * (num_of_neighbor - 1))
            # else:
            #     cluster_coe = 0.0001

            Weight_of_CN[ixtoword[one_news_by_id_word_set_to_list[i]]] = freq_i * NeiContribution
        return Weight_of_CN

    def function_pseg(self, word_pseg_list, word_pseg):
        """
            词性系数
        """
        f = 0.
        if word_pseg in word_pseg_list:
            f = 1.
        return f

    def get_news_title_word_list(self, news_index, news_title_category):
        """

        :param news_index:
        :param news_title_category:
        :param wordtoix:
        :return: 获取编号为news_index的新闻标题词语列表
        """
        news_title_word_list = []
        for word in news_title_category[news_index].split()[1:]:
            news_title_word_list.append(word.encode('utf-8'))
        return news_title_word_list

    def clean_title(self, news_title_word_list, word_flag, word_pseg_list, wordtoix):
        """
            去除标题中没有意义的词
        """
        news_title_word_by_id_list_cleaned = []
        for word in news_title_word_list:
            pseg = word_flag[word]
            if pseg in word_pseg_list:
                news_title_word_by_id_list_cleaned.append(wordtoix[word])
        return news_title_word_by_id_list_cleaned

    def load_useful_word_psegs(self, file_dir):
        """
            加载有用的词性列表
        """
        word_pseg_file = open(file_dir, 'rb')
        word_pseg_list = []
        for line in word_pseg_file:
            word_pseg = line.split()[0]
            word_pseg_list.append(word_pseg)
        return word_pseg_list

    def function_relating(self, V_title, W, word_id):
        """
        :param V_title: 由词语标号表示的新闻标题
        :param W: 词语标号-词向量字典
        :param word_id: 词语ID
        :return: 当前词语基于与新闻标题相关度的权重
        """
        max_relating = 0.1
        arrA = np.asarray(W[word_id])
        for id in V_title:
            arrB = np.asarray(W[id])
            cosine_dist = WeightCompute.CosineDist(self, arrA, arrB)
            if cosine_dist > max_relating:
                max_relating = cosine_dist
        return max_relating

    def count(self, all_news_content_by_id, news_title_category, ID_of_word, category):
        """
        :param all_news_content_by_id:
        :param news_title_category:
        :param ID_of_word:
        :param category:
        :return: 包含词ID_of_word，且类别为category的新闻数量count_X,类别不是category的新闻数量count_P
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

    def CosineDist(self, arrA, arrB):
        """
            功能：cosine距离距离计算
            输入：两个一维数组
        """
        return np.dot(arrA, arrB) / (np.linalg.norm(arrA) * np.linalg.norm(arrB))

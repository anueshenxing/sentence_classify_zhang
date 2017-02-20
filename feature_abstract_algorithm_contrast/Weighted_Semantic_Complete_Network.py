# coding=utf-8
"""
    计算基于语义复杂网络的权值，并排序，保存结果至WSCN.txt文件中
    由于计算一条新闻需要2min左右比较慢，因此选取其中的一部分进行验证
"""
import cPickle
import time
from util.tools import *

reload(sys)
sys.setdefaultencoding("utf-8")


def CosineDist(arrA, arrB):
    """
        功能：cosine距离距离计算
        输入：两个一维数组
        """
    return np.dot(arrA, arrB) / (np.linalg.norm(arrA) * np.linalg.norm(arrB))


if __name__ == "__main__":
    pre_dir = "/home/zhang/PycharmProjects/sentence_classify_zhang/data_file_2017/"

    print "程序开始时间：" + time.asctime(time.localtime(time.time()))

    # 加载文件地址
    wordtoix_and_ixtoword_dir = pre_dir + "wordtoix_and_ixtoword_true.p"
    all_news_content_by_id_dir = pre_dir + "all_news_content_by_id_true.p"
    word_vec_dict_dir = pre_dir + "word_vec_dict_true.p"

    # 加载文件
    wordtoix_and_ixtoword_p = cPickle.load(open(wordtoix_and_ixtoword_dir, "rb"))
    all_news_content_by_id_p = cPickle.load(open(all_news_content_by_id_dir, "rb"))
    word_vec_dict_p = cPickle.load(open(word_vec_dict_dir, "rb"))

    # 获取需要的数据
    wordtoix, ixtoword = wordtoix_and_ixtoword_p[0], wordtoix_and_ixtoword_p[1]
    all_news_content_by_id = all_news_content_by_id_p[0]
    W = word_vec_dict_p[0]  # 保存词的标号 与向量的对应关系

    del wordtoix_and_ixtoword_p, all_news_content_by_id_p, word_vec_dict_p
    print "程序加载数据完毕时间：" + time.asctime(time.localtime(time.time()))

    num_of_all_news = len(all_news_content_by_id)
    for index_of_news in range(1):  # num_of_all_news
        one_news_by_id = all_news_content_by_id[index_of_news]
        one_news_by_id_word_set = set(one_news_by_id)
        one_news_by_id_word_set_to_list = list(one_news_by_id_word_set)
        # 新闻内容中词的个数
        n = len(one_news_by_id_word_set)

        # 两词的语义相似度大于b时，建立连接
        b = 0.5

        # 新闻内容中的词的连接矩阵
        E = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i, n):
                word_i = one_news_by_id_word_set_to_list[i]
                word_j = one_news_by_id_word_set_to_list[j]
                arrA = np.asarray(W[word_i])
                arrB = np.asarray(W[word_j])
                cosine_dist = CosineDist(arrA, arrB)
                if i == j:
                    E[i][j] = 1.
                if cosine_dist > b:
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

            # 计算节点的聚合系数
            cluster_coe = 0.  # 聚类系数
            num_of_neighbor = len(neighbor)
            # 计算邻居节点间的连接数
            for p in range(num_of_neighbor):
                for q in range(p + 1, num_of_neighbor):
                    if E[p][q] > 0.5:
                        cluster_coe += 1
            if len(neighbor) > 1:
                cluster_coe = 2 * cluster_coe / (num_of_neighbor * (num_of_neighbor - 1))
            else:
                cluster_coe = 0.0001

            Weight_of_CN[one_news_by_id_word_set_to_list[i]] = cluster_coe * freq_i * NeiContribution

        Weight_of_CN = sorted(Weight_of_CN.iteritems(), key=lambda d: d[1], reverse=True)
        for i in range(len(Weight_of_CN)):
            print ixtoword[Weight_of_CN[i][0]] + ":" + str(Weight_of_CN[i][1])

    print "程序执行完毕时间：" + time.asctime(time.localtime(time.time()))

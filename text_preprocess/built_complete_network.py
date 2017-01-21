# coding=utf-8
"""
    根据新闻内容中词的语义关系构建复杂网络，G(V, E),其中V是新闻内容中词的集合，E是V中两词之间的边（两词的语义相似度大于b时，建立连接）
"""
import cPickle
from collections import defaultdict
import numpy as np
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


# if __name__ == "__main__":
#     pre_dir = "/home/zhang/PycharmProjects/sentence_classify_zhang/data_file/"
#     word_pseg_dir = pre_dir + "word_flag_lsit.txt"
#     word_pseg_list = load_useful_word_psegs(word_pseg_dir)
#     print function_pseg(word_pseg_list, "n")

if __name__ == "__main__":
    pre_dir = "/home/zhang/PycharmProjects/sentence_classify_zhang/data_file/"
    all_news_word_tf_idf_and_others_dir = pre_dir + "all_news_word_tf_idf_and_others.p"
    word_vec_dict_dir = pre_dir + "word_vec_dict.p"
    word_pseg_dir = pre_dir + "word_flag_lsit.txt"
    word_flag_vocab_dir = pre_dir + "word_flag_vocab.p"
    all_news_title_and_ctg_dir = pre_dir + "news_title_data_and_category.txt"

    all_news_word_tf_idf_and_others = cPickle.load(open(all_news_word_tf_idf_and_others_dir, "rb"))
    word_vec_dict = cPickle.load(open(word_vec_dict_dir, "rb"))
    word_flag_vocab = cPickle.load(open(word_flag_vocab_dir, "rb"))

    wordtoix, ixtoword, all_news_content_by_id, all_news_word_tf_idf = \
        all_news_word_tf_idf_and_others[0], all_news_word_tf_idf_and_others[1], \
        all_news_word_tf_idf_and_others[3], all_news_word_tf_idf_and_others[4]

    word_pseg_list = load_useful_word_psegs(word_pseg_dir)
    all_news_title_by_id = load_all_news_title_by_id(all_news_title_and_ctg_dir, wordtoix)
    word_flag = word_flag_vocab[0]
    all_news_title_by_id_cleaned = clean_title(all_news_title_by_id, word_flag, word_pseg_list, ixtoword)

    index_of_news = 1

    # 保存词的标号 与向量的对应关系
    W = word_vec_dict[0]

    V = all_news_content_by_id[index_of_news]

    # 新闻内容中词的个数
    n = len(V)

    # 两词的语义相似度大于b时，建立连接
    b = 0.5

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

    """
        生成复杂网络数据 版本1
    """
    # webkit_dep = defaultdict()
    # nodes = []
    # links = []
    # categories = []
    #
    # for i in range(n):
    #     node_dict = defaultdict()
    #     node_dict["category"] = 1
    #     node_dict["name"] = ixtoword[V[i]].encode("utf-8")
    #     node_dict["value"] = 1
    #     nodes.append(node_dict)
    #     for j in range(i+1, n):
    #         similarity = E[i][j]
    #         if similarity > 0.5:
    #             relate_dict = defaultdict()
    #             relate_dict["source"] = i
    #             relate_dict["target"] = j
    #             links.append(relate_dict)
    #
    # category = defaultdict()
    # category["name"] = "HTMLElement"
    # category["keywords"] = {}
    # category["base"] = "HTMLElement"
    # category["itemStyle"] = {
    #             "normal": {
    #                 "brushType": "both",
    #                  "color": "#D0D102",
    #                  "strokeColor": "#5182ab",
    #                  "lineWidth": 1
    #             }}
    # categories.append(category)
    # category["name"] = "WebGL"
    # category["keywords"] = {}
    # category["base"] = "WebGLRenderingContext"
    # category["itemStyle"] = {
    #             "normal": {
    #                 "brushType": "both",
    #                  "color": "#00A1CB",
    #                  "strokeColor": "#5182ab",
    #                  "lineWidth": 1
    #             }}
    # categories.append(category)
    #
    # webkit_dep["type"] = "force"
    # webkit_dep["categories"] = categories
    # webkit_dep["nodes"] = nodes
    # webkit_dep["links"] = links
    # with open(pre_dir +'webkei_dep.json','wb') as f:
    #     json.dump(webkit_dep, f)

    """
        生成复杂网络数据版本2
    """
    webkit_dep = defaultdict()
    nodes = []
    links = []

    for i in range(n):
        node_dict = defaultdict()
        node_dict["category"] = 0
        node_dict["name"] = ixtoword[V[i]].encode("utf-8")
        nodes.append(node_dict)
        for j in range(i + 1, n):
            similarity = E[i][j]
            if similarity > 0.5:
                relate_dict = defaultdict()
                relate_dict["source"] = i
                relate_dict["target"] = j
                relate_dict["weight"] = 1
                links.append(relate_dict)

    webkit_dep["nodes"] = nodes
    webkit_dep["links"] = links
    with open(pre_dir + 'complete_network.json', 'wb') as f:
        json.dump(webkit_dep, f)

        # # 计算复杂网络节点重要性指标
        # IMP = defaultdict(float)
        # Vi_NeiContribution = []
        # sumAll_NeiContribution = 0.
        # for i in range(n):
        #     NeiContribution = 0.
        #     for j in range(n):
        #         if i != j and E[i][j] > 0.5:
        #             freq_j = float(all_news_word_tf_idf[index_of_news][ixtoword[V[j]]])
        #             strength_i_j = float(E[i][j])
        #             authority = freq_j * strength_i_j
        #             NeiContribution += authority
        #         else:
        #             NeiContribution += 0.
        #     Vi_NeiContribution.append(NeiContribution)
        #     sumAll_NeiContribution += NeiContribution
        #
        # # 得到基于复杂网络的权重
        # for i in range(n):
        #     IMP[ixtoword[V[i]]] = Vi_NeiContribution[i] / sumAll_NeiContribution
        #
        #
        # # 添加词性系数
        # for i in range(n):
        #     pseg = word_flag[ixtoword[V[i]].decode("utf-8")]
        #     # print ixtoword[V[i]] + ":" + str(pseg)
        #     IMP[ixtoword[V[i]]] *= function_pseg(word_pseg_list, pseg)
        #
        # # 添加与标题相关度系数
        # V_title = all_news_title_by_id_cleaned[index_of_news] #对应的新闻标题
        # for i in range(n):
        #     IMP[ixtoword[V[i]]] *= function_relating(V_title, W, V[i])
        #
        # # 添加tf_idf系数
        # for i in range(n):
        #     IMP[ixtoword[V[i]]] *= all_news_word_tf_idf[index_of_news][ixtoword[V[i]]]
        #
        # IMP = sorted(IMP.iteritems(), key=lambda d: d[1], reverse=True)
        # for i in range(len(IMP)):
        #     print IMP[i][0] + ":" + str(IMP[i][1])

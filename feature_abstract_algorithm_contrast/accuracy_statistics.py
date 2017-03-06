# coding=utf-8
import cPickle
import sys
import time
from collections import defaultdict
import numpy as np
reload(sys)
sys.setdefaultencoding("utf-8")
if __name__ == "__main__":
    pre_dir = "/home/zhang/Desktop/keywords_extract_result/"
    MF_keywords_dir = pre_dir + "MF_keywords.txt"
    complete_network_keywords_dir = pre_dir + "complete_network_keywords.txt"
    tf_crf_keywords_dir = pre_dir + "tf-crf_keywords.txt"
    tf_idf_keywords_dir = pre_dir + "tf-idf_keywords.txt"
    real_keywords_dir = pre_dir + "real_keywords_6.txt"

    algorithms = [MF_keywords_dir, complete_network_keywords_dir, tf_crf_keywords_dir, tf_idf_keywords_dir]
    for dir in algorithms:
        MF_keywords = open(dir, 'rb')
        real_keywords = open(real_keywords_dir, 'rb')
        MF_keywords_dict = defaultdict()
        real_keywords_dict = defaultdict()
        for line in MF_keywords.readlines():
            count = line.split(":")
            if len(count) > 2:
                keywords = []
                news_index = count[1]
                keywords_txt = count[2].split("\n")[0]
                keywords = keywords_txt.split(" ")[:6]
                MF_keywords_dict[news_index] = keywords
        for line in real_keywords.readlines():
            news_index = line.split(":")[0]
            keywords_txt = line.split(":")[1].split("\n")[0]
            keywords = keywords_txt.split(" ")
            real_keywords_dict[news_index] = keywords
        acurracy_list = []
        recall_list = []
        for news_index in real_keywords_dict.keys():
            real_keywords_list = real_keywords_dict[news_index]
            MF_keywords_list = MF_keywords_dict[news_index]
            acurracy = len(list(set(real_keywords_list).intersection(set(MF_keywords_list)))) / float(len(MF_keywords_list))
            recall = len(list(set(real_keywords_list).intersection(set(MF_keywords_list))))/ float(len(real_keywords_list))
            acurracy_list.append(acurracy)
            recall_list.append(recall)

        acurracy_list = np.asarray(acurracy_list)
        recall_list = np.asarray(recall_list)
        F1 = 2*acurracy_list.mean()*recall_list.mean()/(acurracy_list.mean() + recall_list.mean())
        print dir + "->" + str(acurracy_list.mean()) + "->" + str(recall_list.mean()) + str() + "->" + str(F1)



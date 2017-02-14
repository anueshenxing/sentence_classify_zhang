# encoding=utf-8
"""
    功能：根据news_title_data_and_category.txt生成训练数据集和测试数据集，
         并打乱顺序，将结果保存在train_test_data.p中
"""
import random
import cPickle
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

def prepare_labels(labels):
    """
    Process labels to numerical values
    """
    d = {}
    count = 0
    setlabels = set(labels)
    for w in setlabels:
        d[w] = count
        count += 1
    idxlabels = np.array([d[w] for w in labels])
    return idxlabels

if __name__ == "__main__":
    pre_dir = "/home/zhang/PycharmProjects/sentence_classify_zhang/data_file_2017/"
    # news_title_data_and_category_dir = pre_dir + "news_title_data_and_category.txt"
    all_news_title_and_ctg_with_keywords_dir = pre_dir + "all_news_title_and_ctg_with_keywords.txt"
    wordtoix_and_ixtoword_dir = pre_dir + "wordtoix_and_ixtoword_true.p"

    # news_title_data_and_category = open(news_title_data_and_category_dir, 'rb')
    all_news_title_and_ctg_with_keywords = open(all_news_title_and_ctg_with_keywords_dir, 'rb')
    wordtoix_and_ixtoword = cPickle.load(open(wordtoix_and_ixtoword_dir, "rb"))
    wordtoix = wordtoix_and_ixtoword[0]
    # print wordtoix
    del wordtoix_and_ixtoword
    # news_title_category = []
    news_title_category_with_keywords = []
    title = []
    label = []
    train = []
    test = []
    train_label = []
    test_label = []
    for sent in all_news_title_and_ctg_with_keywords:
        news_title_category_with_keywords.append(sent)

    del all_news_title_and_ctg_with_keywords

    random.shuffle(news_title_category_with_keywords)
    for i in range(len(news_title_category_with_keywords)):
        sent_by_id = []
        sent = news_title_category_with_keywords[i]
        word_list = sent.split()
        ctg = word_list[0]
        for j in range(1, len(word_list)):
            sent_by_id.append(wordtoix[word_list[j].encode("utf-8")])

        title.append(sent_by_id)
        label.append(ctg)

    label_by_id = prepare_labels(label)
    # for i in range(len(label_by_id)):
    #     print label[i] + str(label_by_id[i])
    count_per_8 = len(label_by_id)/10*8

    train = title[:count_per_8]
    test = title[count_per_8:]
    train_label = label_by_id[:count_per_8]
    test_label = label_by_id[count_per_8:]
    train_data_label = (train, train_label)
    test_data_label = (test, test_label)

    cPickle.dump([train_data_label, test_data_label], open(pre_dir+"train_test_with_keywords_data.p", "wb"), True)

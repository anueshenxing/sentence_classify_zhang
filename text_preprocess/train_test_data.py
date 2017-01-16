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
    pre_dir = "/home/zhang/PycharmProjects/sentence_classification/data_file/"
    news_title_data_and_category_dir = pre_dir + "news_title_data_and_category.txt"
    all_news_word_tf_idf_and_others_dir = pre_dir + "all_news_word_tf_idf_and_others.p"

    news_title_data_and_category = open(news_title_data_and_category_dir, 'rb')
    all_news_word_tf_idf_and_others = cPickle.load(open(all_news_word_tf_idf_and_others_dir, "rb"))
    wordtoix = all_news_word_tf_idf_and_others[0]
    # print wordtoix
    del all_news_word_tf_idf_and_others
    news_title_category = []
    title = []
    label = []
    train = []
    test = []
    train_label = []
    test_label = []
    for sent in news_title_data_and_category:
        news_title_category.append(sent)

    del news_title_data_and_category

    random.shuffle(news_title_category)
    for i in range(len(news_title_category)):
        sent_by_id = []
        sent = news_title_category[i]
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

    cPickle.dump([train_data_label, test_data_label], open(pre_dir+"train_test_data.p", "wb"))

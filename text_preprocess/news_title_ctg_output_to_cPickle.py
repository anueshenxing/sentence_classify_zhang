# coding=utf-8
"""
    将news_title_data_and_category.txt文件转存为cPickle格式
    将all_news_title_and_ctg_with_keywords.txt文件转存为cPickle格式
"""
import cPickle
import sys

reload(sys)
sys.setdefaultencoding("utf-8")

# if __name__ == '__main__':
#     pre_dir = "/home/zhang/PycharmProjects/sentence_classify_zhang/data_file_2017/"
#     news_title_data_and_category_dir = pre_dir + "news_title_data_and_category.txt"
#     news_title_data_and_category = open(news_title_data_and_category_dir, 'rb')
#     news_title_category = []
#     for sent in news_title_data_and_category:
#         news_title_category.append(sent)
#
#     cPickle.dump([news_title_category], open(pre_dir + "news_title_category.p", "wb"), True)

if __name__ == '__main__':
    pre_dir = "/home/zhang/PycharmProjects/sentence_classify_zhang/data_file_2017/"
    all_news_title_and_ctg_with_keywords_dir = pre_dir + "all_news_title_and_ctg_with_keywords.txt"
    all_news_title_and_ctg_with_keywords = open(all_news_title_and_ctg_with_keywords_dir, 'rb')
    news_title_category_with_keywords = []
    for sent in all_news_title_and_ctg_with_keywords:
        news_title_category_with_keywords.append(sent)

    cPickle.dump([news_title_category_with_keywords], open(pre_dir + "news_title_category_with_keywords.p", "wb"), True)

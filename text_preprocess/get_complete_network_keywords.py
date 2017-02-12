# encoding=utf-8
"""
    将基于多特征融合提取到的关键词保存到 complete_network_keywords.p 中,
    complete_network_keywords.p 中以两种形式存储，第一种是以字符串形式，第二种是以数组形式,
    每一行的词的顺序即是由大到小的关键词重要性顺序
"""
import cPickle
import sys

reload(sys)
sys.setdefaultencoding("utf-8")

if __name__ == '__main__':
    pre_dir = "/home/zhang/PycharmProjects/sentence_classify_zhang/data_file_2017/"
    news_title_category_dir = pre_dir + "news_title_category.p"
    news_title_category_p = cPickle.load(open(news_title_category_dir, "rb"))
    news_title_category = news_title_category_p[0]

    news_title_category_with_keywords_dir = pre_dir + "news_title_category_with_keywords.p"
    news_title_category_with_keywords_p = cPickle.load(open(news_title_category_with_keywords_dir, "rb"))
    news_title_category_with_keywords = news_title_category_with_keywords_p[0]

    complete_network_keywords_stored_by_String = []
    complete_network_keywords_stored_by_List = []

    for i in range(len(news_title_category)):
        if i % 1000 == 0:
            print i

        title_with_keywords = news_title_category_with_keywords[i]
        title = news_title_category[i]
        title_with_keywords_word_list = title_with_keywords.split()
        title_word_list = title.split()
        keywords_word_list = []

        for word in title_with_keywords_word_list:
            if word not in title_word_list:
                keywords_word_list.append(word)

        keywords = " ".join(keywords_word_list)

        complete_network_keywords_stored_by_String.append(keywords)
        complete_network_keywords_stored_by_List.append(keywords_word_list)
        # print "title_with_keywords: ->" + title_with_keywords
        # print "title: ->" + title
        # print "提取到的关键词： ->" + keywords
        # print "----------------------------"

    cPickle.dump([complete_network_keywords_stored_by_String, complete_network_keywords_stored_by_List],
                 open(pre_dir + "complete_network_keywords.p", "wb"), True)

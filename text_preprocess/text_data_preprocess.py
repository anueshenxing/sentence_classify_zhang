# encoding=utf-8
"""
    生成 词向量训练语料：yuliao_20170208
        词-词性对照表：word_flag_vocab
        新闻标题-类别数据集：news_title_data_and_category
        新闻内容分词结果：news_content_fenci_data
"""
from collections import defaultdict

import cPickle
import jieba
import jieba.posseg as pseg
from pymongo import *
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

def jieba_fenci(text):
    term = text.split('\n')
    s = ""
    for i in term:
        s += i
    # 仅生成分词
    # seg_list = jieba.cut(s)
    # return seg_list

    # 生成分词及其词性
    word_flag_dict = pseg.cut(s)
    return word_flag_dict

def stopwords(file_dir):
    f = open(file_dir, 'rb')
    stopwords = []
    for word in f:
        word = word.split('\n')[0]
        stopwords.append(word.encode('utf-8'))
    return stopwords

if __name__ == '__main__':
    client = MongoClient("localhost", 27017)
    db = client.news_db
    news = db.news_collection
    all_news = news.find(no_cursor_timeout=True)
    count = 0
    pre_dir = "/home/zhang/PycharmProjects/sentence_classify_zhang/data_file_2017/"
    yuliao_20170208_dir = pre_dir + "yuliao_20170208.txt"
    stopwords_file_dir = pre_dir + "stopwords_csdn_shijieba2009.txt"
    news_title_data_and_category_dir = pre_dir + "news_title_data_and_category.txt"
    news_content_fenci_data_dir = pre_dir + "news_content_fenci_data.txt"

    yuliao_20170208 = open(yuliao_20170208_dir, "a")
    news_title_data_and_category = open(news_title_data_and_category_dir, "a")
    news_content_fenci_data = open(news_content_fenci_data_dir, "a")
    stopwords = stopwords(stopwords_file_dir)
    word_flag_vocab = defaultdict(float)


    for one_news in all_news:
        print count
        count += 1
        news_ctg = one_news['news_ctg'].decode('utf-8')
        news_title = one_news['news_title'].decode('utf-8')
        news_content = one_news['news_content'].decode('utf-8')
        ctg_title_txt = ""
        one_news_text = ""
        title = []
        content = []
        result = []
        for word, flag in jieba_fenci(news_title):
            if word not in stopwords:
                word_flag_vocab[word.encode("utf-8")] = flag
                title.append(word)
                result.append(word)
                # print word

        for word, flag in jieba_fenci(news_content):
            if word not in stopwords:
                word_flag_vocab[word.encode("utf-8")] = flag
                result.append(word)
                content.append(word)
                # print word

        ctg_title_txt = news_ctg + " " + " ".join(title) + "\n"
        one_news_text = " ".join(result) + "\n"
        content_txt = " ".join(content) + "\n"
        ctg_title_txt = ctg_title_txt.encode("utf-8")
        one_news_text = one_news_text.encode("utf-8")
        content_txt = content_txt.encode("utf-8")
        news_title_data_and_category.write(ctg_title_txt)
        yuliao_20170208.write(one_news_text)
        news_content_fenci_data.write(content_txt)
    all_news.close()
    cPickle.dump([word_flag_vocab], open(pre_dir + "word_flag_vocab.p", "wb"))

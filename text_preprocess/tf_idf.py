# coding=utf-8
"""
    生成 词-id、id-词对照表：wordtoix, ixtoword
        词-词的IDF对照表、新闻内容的词id表示：word_IDF, all_news_content_by_id
        词-词的tf_idf对照表：word_tf_idf
"""
import cPickle

import time

from util.tools import *

reload(sys)
sys.setdefaultencoding("utf-8")

if __name__=="__main__":
    pre_dir = "/home/zhang/PycharmProjects/sentence_classify_zhang/data_file_2017/"
    yuliao_20170208_dir = pre_dir + "yuliao_20170208.txt"
    news_content_fenci_data_dir = pre_dir + "news_content_fenci_data.txt"
    count = 0
    news_title_and_content_data = open(yuliao_20170208_dir, 'rb')
    news_content_fenci_data = open(news_content_fenci_data_dir, 'rb')

    all_news_title_and_content = [] #保存所有新闻的标题和内容，都经过分词、去停用词
    all_news_content_by_id = [] #保存所有新闻的内容,其中每个词都用词的id表示
    vocab_statitic = defaultdict(float) #保存词的IDF
    all_news_word_tf_idf = [] #保存每一条新闻中每个词的tf_idf值

    print time.asctime(time.localtime(time.time()))
    for sent in news_title_and_content_data:
        sent = sent.split("\n")[0]
        all_news_title_and_content.append(sent)

    wordtoix, ixtoword = build_vocab(all_news_title_and_content)
    print "wordtoix, ixtoword already be built!"

    print time.asctime(time.localtime(time.time()))
    # 将新闻内容的词用id表示
    for sent in news_content_fenci_data:
        sent = sent.split("\n")[0]
        sent_by_id = []
        for word in sent.split():
            sent_by_id.append(wordtoix[word])
        all_news_content_by_id.append(sent_by_id)
    print "all_news_content_by_id already be built!"
    print time.asctime(time.localtime(time.time()))

    # 统计每个词在所有文档中出现的次数
    for id_sent in all_news_content_by_id:
        id_sent_set = set(id_sent)
        for id in id_sent_set:
            vocab_statitic[id] += 1.
    #文档总数
    N = len(all_news_content_by_id)

    # 将文档出现次数转换为IDF
    for id in vocab_statitic.keys():
        vocab_statitic[id] = math.log(N/vocab_statitic[id])

    # 计算每一条新闻中每个词的tf_idf值
    print "start to bulid all_news_word_tf_idf!"
    for one_news_content in all_news_content_by_id:
        if count % 1000 == 0:
            print count
        count += 1
        word_tf_idf = defaultdict(float)
        n = float(len(one_news_content))
        for i in set(one_news_content):
            tf_idf = one_news_content.count(i) / n * vocab_statitic[i]
            word_tf_idf[ixtoword[i]] = round(tf_idf, 10)
        all_news_word_tf_idf.append(word_tf_idf)

    print time.asctime(time.localtime(time.time()))
    # 保存数据
    # cPickle.dump([wordtoix, ixtoword, all_news_content, all_news_content_by_id, all_news_word_tf_idf],
    #              open(pre_dir + "all_news_word_tf_idf_and_others.p", "wb"))
    cPickle.dump([wordtoix, ixtoword],
                 open(pre_dir + "wordtoix_and_ixtoword_true.p", "wb"), True)
    cPickle.dump([all_news_content_by_id],
                 open(pre_dir + "all_news_content_by_id_true.p", "wb"), True)
    cPickle.dump([all_news_word_tf_idf],
                 open(pre_dir + "all_news_word_tf_idf_true.p", "wb"), True)
    print time.asctime(time.localtime(time.time()))

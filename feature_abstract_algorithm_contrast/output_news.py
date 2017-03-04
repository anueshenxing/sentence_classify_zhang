# coding=utf-8
import sys
from pymongo import *

reload(sys)
sys.setdefaultencoding("utf-8")

if __name__ == "__main__":
    pre_dir = "/home/zhang/news_data/"
    client = MongoClient("localhost", 27017)
    db = client.news_db
    news = db.news_collection
    all_news = news.find(no_cursor_timeout=True)
    count = 0
    current_dir = ''
    news_file = ''
    for one_news in all_news:
        if count % 5000 == 0:
            current_dir = pre_dir + str(count) + ".txt"
            news_file = open(current_dir, 'a')
            print "当前文件存储位置为：" + current_dir
        news_ctg = one_news['news_ctg'].decode('utf-8')
        news_title = one_news['news_title'].decode('utf-8')
        news_content = one_news['news_content'].decode('utf-8')

        one_news = "新闻ID: " + str(count) + " || " + "category: " + news_ctg + " || title: " + news_title+\
                   " || content: " + news_content + "\n"
        news_file.write(one_news)
        count += 1

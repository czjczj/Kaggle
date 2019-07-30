#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/7/25 14:49
#@Author: czj
#@File  : news_word2vec.py

import pandas as pd
import numpy as np
import os
import re
# path = "D:/MyInstallData/PyCharm/Kaggle/sogo/"
path = "./"
f = open(path+"zh_stopword.txt",'r',encoding='utf-8')
stopwords = [i.strip() for i in f]
# a = '<content>环球网记者李亮报道，正在意大利度蜜月的“脸谱”</content>\n'
# b = '<contenttitle>扎克伯格携妻罗马当街吃３０元麦当劳午餐（组图）</contenttitle>\n'
# c = '<doc>\n'
def load_txt(path):
    import re
    f = open(path, encoding='utf-8')
    titles = []
    contents = []
    new_count = 0
    for line in f:
        title = re.findall('<contenttitle>(.*?)</contenttitle>', line.strip())
        if title != []:
            titles.extend(title)
        content = re.findall('<content>(.*?)</content>', line.strip())
        if content != []:
            contents.extend(content)
    return contents, titles
def preprocessing(texts):
    import jieba.posseg as psg
    #1. 分词
    def get_words(text):
        words = []
        seg = psg.cut(text)
        for ele in seg:
            if ele.flag != 'v' and ele.flag != 'd' and ele.flag != 't':
                words.append(ele.word)
        return words
    text2word = [get_words(text) for text in texts]
    # print(text2word)
    #2. 去掉数字，字符, 停顿词
    def word_filter(words):
        rtn = []
        for word in words:
            if bool(re.search(r"[^a-zA-Z]\d+", word)):
                continue
            if word in stopwords:
                continue
            if bool(re.search(r"\\ue",word)):
                continue
            if len(word) == 1:
                continue
            rtn.append(word)
        return rtn
    text2word = [word_filter(words) for words in text2word]
    return text2word

def get_Word2Vec(corpus, model_path):
    import multiprocessing
    from gensim.models.word2vec import Word2Vec
    model = Word2Vec(corpus, size=128, window=3, min_count=3, workers=multiprocessing.cpu_count())
    model.save(model_path)

if __name__=='__main__':
    # contents, titles = load_txt(path+"sogoSmall.txt")
    contents, titles = load_txt(path+"corpus.txt")
    word_vec = preprocessing(contents)

    # model_path = path+"sogo_small_contents.model"
    model_path = path+"sogo_big_contents.model"
    model = Word2Vec.load(os.path.join(path,'sogo_big_contents.model'))

    print(model.most_similar(u"北京"))
    print(model.most_similar(u"记者"))
    print(model.most_similar(u"中国"))


# from gensim.models.word2vec import Word2Vec
# model = Word2Vec.load(path+ "sogo_small_contents.model")
# model.most_similar(u"北京")
# model.most_similar(u"记者")
# model.most_similar(u"中国")

# import re
# a = "4123"
# decimal_regex = re.compile(r"[^a-zA-Z]\d+")
# res = decimal_regex.sub(r"",a)

# a = list(jieba.cut(contents[2]))
# b = word_filter(a)

# from nltk.stem import WordNetLemmatizer
# wordNet = WordNetLemmatizer()
# wordNet.lemmatize('dogs')
# wordNet.lemmatize("went", pos='v')

# #
# import jieba.posseg as psg
#
# text = "目前现在"
# #词性标注
# seg = psg.cut(text)
#
# #将词性标注结果打印出来
# for ele in seg:
#     print(ele.word, ele.flag, type(ele.word))


# import logging
# logger = logging.getLogger('czj')
# logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
# logging.root.setLevel(level=logging.INFO)

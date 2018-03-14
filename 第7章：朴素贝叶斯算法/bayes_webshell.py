# -*- encoding:utf-8 -*-
import os
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
import codecs

# |表示前面和后面的 或 匹配
r_token_pattren = r'\b\w+\b\(|\'\w+\'' 

def load_file(file_path):
    t = ''
    with codecs.open(file_path, "r") as f:
        for line in f:
            line = line.strip("\n")
            t += line
    return t

def load_files(path):
    file_list = []
    #os.walk，遍历该目录下面所有的文件，目录等
    for r, d, files in os.walk(path):
        for file in files:
            if file.endswith('.php'):
                file_path = path + file
                t = load_file(file_path)
                file_list.append(t)
    return file_list

if __name__ == "__main__":
    #参数分析：ngram_range:抽取gram的方式，decode_error编码报错，token_pattren：选择文档， min_df频数小于1的情况下不计考虑
    webshell_bigram_vectorizer = CountVectorizer(ngram_range=(2, 2), decode_error='ignore',
                                        token_pattern=r'\b\w+\b', min_df=1)
    webshell_files_list = load_files("/home/ren/Desktop/machine_learning/data/1book/data/PHP-WEBSHELL/xiaoma/")
    x1 = webshell_bigram_vectorizer.fit_transform(webshell_files_list).toarray()
    y1 = [1]*len(x1)
    vocabulary = webshell_bigram_vectorizer.vocabulary_

    #CountVectorizer()是将文本向量转换为字符频率向量
    wp_bigram_vectorizer = CountVectorizer(ngram_range=(2, 2), decode_error='ignore', token_pattern=r"(?u)\b\w\w+\b", min_df=1, vocabulary=vocabulary)
    wp_files_list = load_files("/home/ren/Desktop/machine_learning/data/1book/data/wordpress/")
    x2 = wp_bigram_vectorizer.fit_transform(wp_files_list).toarray()
    y2 = [0]*len(x2)
    # print(webshell_bigram_vectorizer.vocabulary_)

    x = np.concatenate((x1,x2))
    y = np.concatenate((y1,y2))
    clf = GaussianNB()
    print("11")
    print(cross_validation.cross_val_score(clf, x, y, n_jobs=1, cv=3))
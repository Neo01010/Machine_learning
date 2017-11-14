# -*- encoding:utf-8 -*-

import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
import numpy as np

def load_one_file(filename):
    x=[]
    with open(filename, "r") as f:
        line = f.readline()
        line = line.strip('\n')

    return line

def load_adfa_training_files(rootdir):
    x=[]
    y=[]
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isfile(path):
            x.append(load_one_file(path))
            y.append(0)
    return x, y

def dirlist(path, allfile):
    filelist = os.listdir(path)

    for filename in filelist:
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            dirlist(filepath, allfile)
        else:
            allfile.append(filepath)
    return allfile

def load_adfa_webshell_files(rootdir):
    x = []
    y = []
    allfile = dirlist(rootdir, [])
    for file in allfile:
        if re.match(r"./data/1book/data/ADFA-LD/Attack_Data_Master/Web_Shell_\d+/UAD-W*", file):
            x.append(load_one_file(file))
            y.append(1)
    return x, y

x1, y1 = load_adfa_training_files("./data/1book/data/ADFA-LD/Training_Data_Master/")
x2, y2 = load_adfa_webshell_files("./data/1book/data/ADFA-LD/Attack_Data_Master/")

x = x1+x2
y = y1+y2

#文本x的特征提取以及向量化 min_df表示忽略样本的条件，1表示不忽略任意一条
vectorizer = CountVectorizer(min_df=1)
x = vectorizer.fit_transform(x)
x = x.toarray()
print(y)
clf = KNeighborsClassifier(n_neighbors=3)
scores = cross_validation.cross_val_score(clf, x, y, n_jobs=-1, cv=10)
print(scores)
print(np.mean(scores))

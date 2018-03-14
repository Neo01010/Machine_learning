# -*- encoding:utf-8 -*-
#恶意的标记为1，正常样本的标记为0
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import cross_validation
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pydotplus

def load_one_file(filename):
    x = []
    with open(filename) as f:
        line = f.readline()
        line = line.strip('\n')
    return line

def load_adfa_traing_files(rootdir):
    x = []
    y = []
    files = os.listdir(rootdir)
    for i in files:
        path = os.path.join(rootdir, i)
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

def load_adfa_hydra_ftp_files(rootdir):
    x = []
    y = []
    allfile = dirlist(rootdir, [])
    for file in allfile:
        if re.match(r"./data/1book/data/ADFA-LD/Attack_Data_Master/Hydra_FTP_\d+/UAD-Hydra-FTP*",file):
            x.append(load_one_file(file))
            y.append(1)
    return x, y

if __name__ == "__main__":
    x1, y1 = load_adfa_traing_files("./data/1book/data/ADFA-LD/Training_Data_Master/")
    x2, y2 = load_adfa_hydra_ftp_files("./data/1book/data/ADFA-LD/Attack_Data_Master/")
    x = x1 + x2
    y = y1 + y2
    #进行特征选择，数据处理并进行你和数据
    vectorizer = CountVectorizer(min_df=1)
    x = vectorizer.fit_transform(x)
    #将x转换为ndarry格式
    x = x.toarray()


    #决策数和随机森林的区别
    clf1 = tree.DecisionTreeClassifier()
    score = cross_validation.cross_val_score(clf1, x, y, n_jobs=-1, cv = 10)
    print(type(score))
    print(score)
    print(np.mean(score))
    #可视化
    clf1 = clf1.fit(x, y)
    dot_data = tree.export_graphviz(clf1, out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("./photo/6/ftp.pdf")
    clf2 = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state= 0)
    score = cross_validation.cross_val_score(clf2, x, y, n_jobs=-1, cv=10)
    print(np.mean(score))
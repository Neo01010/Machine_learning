import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
from sklearn import cross_validation
import re
import numpy as np

def load_one_file(filename):
    with open(filename) as f:
        return f.read().strip("\n")
    

def load_adfa_training_files(dirname):
    x = []
    y = []
    file_list = os.listdir(dirname)
    for each_file in file_list:
        path = dirname + each_file
        x.append(load_one_file(path))
        y.append(0)
    return x, y


def load_adfa_java_files(dirname):
    allfile = []
    dir_path = os.listdir(dirname)
    for d in dir_path:
        file_path = os.path.join(dirname, d)
        if re.match(r"/home/ren/Desktop/machine_learning/data/1book/data/ADFA-LD/Attack_Data_Master/Java_Meterpreter_\d+", file_path):
            files = os.listdir(file_path)
            for i in files:
                allfile.append(os.path.join(file_path, i))
    print(allfile)
    x = []
    y = []
    for file in allfile:
        x.append(load_one_file(file))
        y.append(1)
    return x, y

if __name__ == "__main__":
    x1, y1 = load_adfa_training_files("/home/ren/Desktop/machine_learning/data/1book/data/ADFA-LD/Training_Data_Master/")
    x2, y2 = load_adfa_java_files("/home/ren/Desktop/machine_learning/data/1book/data/ADFA-LD/Attack_Data_Master/")

    x = x1 + x2
    y = y1 + y2
    #对原数据提取特征等操作，CounterVectorizer讲文本向量转换为字符频率向量
    vectorizer = CountVectorizer(min_df=1)
    x = vectorizer.fit_transform(x)
    print(type(x))
    x=x.toarray()

    logreg = linear_model.LogisticRegression(C=1e5)
    score = cross_validation.cross_val_score(logreg, x, y, n_jobs=1, cv=10)
    print(np.mean(score))
# -*- encoding:utf-8 -*-

from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report
from sklearn import metrics
import numpy as np
#测试样本数
N = 100

def load_user_cmd(filename):
    """
    提取用户日志操作记录
    """
    cmd_list = []
    dist_max = []
    dist_min = []
    dist = []
    with open(filename, "r") as f:
        i = 0
        x = []
        for line in f:
            line = line.strip("\n")
            x.append(line)
            dist.append(line)
            i += 1
            if i == 100:
                cmd_list.append(x)
                x = []
                i = 0

    fdist = list(FreqDist(dist).keys())
    dist_max = set(fdist[:50])
    dist_min = set(fdist[-50:])
    return cmd_list, dist_max, dist_min

def get_user_cmd_feature(user_cmd_list, dist_max, dist_min):
    """
    利用提取的操作参数序列和最常用命令以及最不常用命令统计用户操作特征
    """
    user_cmd_feature = []
    for cmd_block in user_cmd_list:
        f1 = len(set(cmd_block))
        fdist = list(FreqDist(cmd_block).keys())
        f2 = set(fdist[:10])
        f3 = set(fdist[-10:])
        f2 = len(f2 & dist_max)
        f3 = len(f3 & dist_min)
        x = [f1,f2,f3]
        user_cmd_feature.append(x)
    return user_cmd_feature

def get_label(filename, index=0):
    x = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip("\n")
            x.append(int(line.split()[index]))
    return x

if __name__ == '__main__':
    user_cmd_list, user_cmd_dist_max, user_cmd_dist_min = load_user_cmd("/home/ren/Desktop/machine_learning/data/1book/data/MasqueradeDat/User8")
    user_cmd_feature = get_user_cmd_feature(user_cmd_list, user_cmd_dist_max, user_cmd_dist_min)
    labels = get_label("/home/ren/Desktop/machine_learning/data/1book/data/MasqueradeDat/label.txt",7)
    y = [0]*50 + labels

    x_train = user_cmd_feature[0:N]
    y_train = y[0:N]

    x_test = user_cmd_feature[N:150]
    y_test = y[N:150]

    #下面开始进行训练
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(x_train, y_train)
    y_predict = neigh.predict(x_test)

    score = np.mean(y_test == y_predict)*100

    print(y_test)
    print(y_predict)
    print(score)

    print(classification_report(y_test, y_predict))
    print(metrics.confusion_matrix(y_test, y_predict))



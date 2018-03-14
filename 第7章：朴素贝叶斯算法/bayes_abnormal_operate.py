import os
from nltk.probability import FreqDist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np


def load_user_cmd_new(filename):
    cmd_list = []
    dist = []
    with open(filename) as f:    
        i = 0
        x = []
        for line in f:
            line= line.strip("\n")
            x.append(line)
            dist.append(line)
            i += 1
            if i == 100:
                cmd_list.append(x)
                x = []
                i = 0
    # FreqDist继承dict，统计单词和次数
    fdisk = FreqDist(dist).keys()
    print(type(fdisk))
    return cmd_list, list(fdisk)

def get_user_cmd_feature_new(user_cmd_list, dist):
    user_cmd_feature = []
    for cmd_list in user_cmd_list:
        v = [0]*len(dist)
        for i in range(len(dist)):
            if dist[i] in cmd_list:
                v[i] += 1
        user_cmd_feature.append(v)
    return user_cmd_feature

def get_label(filename, index= 0):
    x = []
    with open(filename) as f:
        for line in f:
            line = line.strip("\n")
            x.append( int(line.split()[index]))

    return x


if __name__ == "__main__":
    user_cmd_list, dist = load_user_cmd_new("/home/ren/Desktop/machine_learning/data/1book/data/MasqueradeDat/User3")
    user_cmd_feature = get_user_cmd_feature_new(user_cmd_list, dist)
    labels = get_label("/home/ren/Desktop/machine_learning/data/1book/data/MasqueradeDat/label.txt",2)
    y = [0]*50 + labels

    x_train = user_cmd_feature[0:90]
    y_train = y[:90]

    x_test = user_cmd_feature[90:150]
    y_test = y[90:150]

    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(x_train, y_train)
    y_predict_knn = neigh.predict(x_test)

    clf = GaussianNB().fit(x_train, y_train)
    y_predict_nb = clf.predict(x_test)

    print("KNN %d"%(np.mean(y_test == y_predict_knn)*100))
    print("NB %d"%(np.mean(y_test == y_predict_nb)*100))
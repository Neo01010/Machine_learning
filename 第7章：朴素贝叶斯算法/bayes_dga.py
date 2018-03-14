#检测域名生成算法（DGA），利用nb算法实现对正常域名和DGA域名进行区分

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
import os
import csv

#要求域名的最小长度
MIN_LEN = 10


def load_alexa(filename):
    domain_list = []
    csv_reader = csv.reader(open(filename))
    for row in csv_reader:
        domain = row[1]
        if len(domain) >= MIN_LEN:
            domain_list.append(domain)
    return domain_list

def load_dga(filename):
    domain_list = []
    with open(filename) as f:
        for line in f:
            domain = line.split(",")[0]
            if len(domain) >= MIN_LEN:
                domain_list.append(domain)
    return domain_list
    



def nb_dga():
    #加载alexa前1000的域名作为白样本，标记为0
    x1domain_list = load_alexa("/home/ren/Desktop/machine_learning/data/1book/data/top-1000.csv")
    #加载cryptolocker的DGA域名，标记为2
    x2crypto_list = load_dga("/home/ren/Desktop/machine_learning/data/1book/data/dga-cryptolocke-1000.txt")
    #加载post-tovar-goz的DGA域名，标记为3
    x3post_list = load_dga("/home/ren/Desktop/machine_learning/data/1book/data/dga-post-tovar-goz-1000.txt")

    x_domain_list = np.concatenate((x1domain_list, x2crypto_list, x3post_list))

    y1 = [0]* len(x1domain_list)
    y2 = [1]* len(x2crypto_list)
    y3 = [2]* len(x3post_list)

    #np.concatenate(()/[], axis=)
    '''
     传入的参数必须是一个 多个数组的元祖或列表
     axis：指定拼接的方向，asix=0表示直接顺序凭借，asix=1表示对应行拼接
    '''

    y = np.concatenate((y1, y2, y3))

    cv = CountVectorizer(ngram_range=(2, 2), decode_error='ignore', token_pattern=r"\w", min_df=1)

    x = cv.fit_transform(x_domain_list).toarray()

    clf = GaussianNB()
    print(cross_validation.cross_val_score(clf, x, y, n_jobs=1, cv=3))


if __name__ == "__main__":
    nb_dga()
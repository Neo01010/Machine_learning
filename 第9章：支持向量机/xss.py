from sklearn import svm
import re
from sklearn import cross_validation
import numpy as np


x = []
y = []

#特征提取
def get_len(url):
    return len(url)

def get_url_count(url):
    #re.IGNORECASE忽略大小写
    if re.search("(http://)|(https://)", url, re.IGNORECASE):
        return 1
    else:
        return 0

def get_evil_char(url):
    return len(re.findall("[<>,\'\"/]", url, re.IGNORECASE))

def get_evil_word(url):
    return len(re.findall("(alert)|(script=)(%3c)|(%3e)|(%20)|(onerror)|(onload)|(eval)|(src=)|(prompt)",url,re.IGNORECASE))

def get_last_char(url):
    if re.search('/$', url, re.IGNORECASE):
        return 1
    else:
        return 0
def get_feature(url):
    return [get_len(url),get_url_count(url),get_evil_char(url),get_evil_word(url),get_last_char(url)]

def etl(filename,data,isxss):
        with open(filename) as f:
            for line in f:
                f1=get_len(line)
                f2=get_url_count(line)
                f3=get_evil_char(line)
                f4=get_evil_word(line)
                data.append([f1,f2,f3,f4])
                if isxss:
                    y.append(1)
                else:
                    y.append(0)
        return data

etl('/home/ren/Desktop/machine_learning/data/1book/data/xss-200000.txt',x,1)
etl('/home/ren/Desktop/machine_learning/data/1book/data/good-xss-200000.txt',x,0)



x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y, test_size=0.4, random_state=0)

clf = svm.SVC(kernel='linear', C=1).fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(np.mean(y_pred == y_test))
print(cross_validation.cross_val_score(clf, x_test, y_test))

# -*- encoding:utf-8 -*-

#效果验证，交叉验证，以svm为例
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score

#iris为bunch类型，
iris = datasets.load_iris()
print(type(iris))
print(iris.data.shape, iris.target.shape)
#利用train_test_split分割随机数据样本
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)
#调用svm进行训练,SVC是svm的一种，SVR是用来做回归的
clf = svm.SVC(kernel='linear', C=1.0).fit(X_train, y_train)
print(clf.score(X_test, y_test))

#k折交叉验证：分为k分，取一份作为测试集。交叉验证重复k次，最后取平均值
scores = cross_val_score(clf, iris.data, iris.target, cv=5)
print(scores)

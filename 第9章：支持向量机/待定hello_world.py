import numpy as np 
import matplotlib.pyplot as plt
from sklearn import svm

#创建40个随机点
#设置瑞吉算法开始的整数值为0
np.random.seed(0)
#np.random.rand()返回0-1之间的值，np.random.randn()返回一组样本，具有标准正太分布
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
print(X)
y = [0] * 20 + [1]*20

#SVC是svm的一种类型，用来做分类的，SVR也是svm一种，用来做回归的
clf = svm.SVC(kernel="linear")
clf.fit(x, y)

#构造超平面：
w = clf.coef_[0]
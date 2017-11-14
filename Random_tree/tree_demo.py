# -*- encoding:utf-8 -*-

#决策树模型常用与数据分类以及预测
from sklearn import tree

x = [[0,0],[1,1]]
y = [0,1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x, y)
print(clf.predict([[2., 2.]]))

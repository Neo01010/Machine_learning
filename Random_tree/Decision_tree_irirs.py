# -*- encoding:utf-8 -*-

from sklearn.datasets import load_iris
from sklearn import tree
#pydotplus用于图形化dot语言接口
import pydotplus

iris = load_iris()
print(type(iris))
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)
dot_data = tree.export_graphviz(clf, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("./photo/6/iris.pdf")
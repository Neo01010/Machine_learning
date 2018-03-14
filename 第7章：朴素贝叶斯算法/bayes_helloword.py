from sklearn import datasets
iris = datasets.load_iris()
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
print(type(iris.data))
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
print(iris.target != y_pred)
#iris.shape[0]，获取数组的行数
print("%d,%d"%(iris.data.shape[0],(iris.target != y_pred).sum()))
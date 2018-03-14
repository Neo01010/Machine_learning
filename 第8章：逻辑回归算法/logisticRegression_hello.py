
import numpy as np 
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
x = iris.data[:, :2]
y = iris.target
h = 0.02
#C是正则项因子，c越大，对模型复杂度惩罚越大，对拟合数据的损失惩罚小，就不会出现过拟合
logreg = linear_model.LogisticRegression(C=1e5)

logreg.fit(x[:], y[:])

#模拟生成一些数据，x代表的是花瓣第一个数据，y代表的是花瓣第二个数据
x_min, x_max = x[:,0].min() - 0.5, x[:,0].max() + .5
y_min, y_max = x[:,1].min() - 0.5, x[:,1].max() + .5

#meshgrid将想x_min变为xx的行向量，x_max变为yy的列向量
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

#ravel（）将多维数组转变为一维,np.c_[a,b]将切片对象延第二个轴转换为连接（按列）
z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

#将结果可视化
z = z.reshape(xx.shape)
#figure()新建绘画窗口，独立显示绘画的图片：1d代表序列化，figsize:初始化大小
plt.figure(1, figsize=(4, 3))
#pcolormesh对坐标点着色，Z代表作色方案，cmap作色配置
plt.pcolormesh(xx, yy, z, cmap=plt.cm.Paired)
#画点点，
plt.scatter(x[:,0], x[:,1], c=y, edgecolors="k", cmap=plt.cm.Paired)
plt.xlabel('sepal length')
plt.ylabel('sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

# plt.show()

z = logreg.predict(x[100:])
print(z)
print(y[100:])
print(np.mean(z == y[100:]))

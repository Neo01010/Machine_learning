import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier

mnist = fetch_mldata("MNIST original", data_home="/home/ren/machine-learning/lib/python3.5/site-packages/sklearn/datasets")
# print(type(mnist), mnist)
print(type(mnist.data))
# mnist.data/255:是为了将值进行 归一化
X, y = mnist.data/ 255. , mnist.target
x_train, x_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]


mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                    solver="sgd", verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)

mlp.fit(x_train, y_train)

fig, axes = plt.subplots(4, 4)

#coefs_:代表权重矩阵
vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
#.T:表示数组的转置
for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin,
               vmax=.5 * vmax)
    ax.set_xticks([0,5,10,20])
    ax.set_yticks(())

plt.show()
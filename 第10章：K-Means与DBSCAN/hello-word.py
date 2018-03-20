import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

def show_kmeans():
    plt.figure(figsize=(12, 12))
    #生成样本总数
    n_samplts = 1500
    #随机生成器的初始化值
    random_state = 170
    #x:返回的样本，Y：返回的特征值
    x, y = make_blobs(n_samples=n_samplts, random_state=random_state)
    print(x[:10], y[:10])
    y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(x)
    plt.subplot(221)
    plt.scatter(x[:,0], x[:,1], c=y_pred)
    plt.title("hello,world")
    plt.show()


if __name__ == "__main__":
    show_kmeans()
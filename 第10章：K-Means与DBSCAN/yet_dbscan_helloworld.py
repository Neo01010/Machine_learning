import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
#StandaraScaler:标准化，由于统一特征不同样本时，取值差别很大，会有所影响
from sklearn.preprocessing import StandardScaler
import numpy as np

def show_dbscan():
    centers = [[1,1], [-1,-1], [1, -1]]
    #生成高斯分布的聚类数据，n_samples:生成样本数量 n_features:样本特征数 centers：确定的中心点 cluster_std：每个类别的方差 random_state：随机初始化参数
    x, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4, random_state=0)
    print(x)
    x = StandardScaler().fit_transform(x)
    print(x)
    #eps:同一聚类集合中任意两个样本的最大距离, min_samples:同一聚类中的最小样本数 algorithm:使用的算法 n_jobs:并发任务数
    db = DBSCAN(eps=0.5, min_samples=10).fit(x)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    print(core_samples_mask)
    #core_sample_indices_:核心样本指数
    # db.core_sample_indices_:表示的是某个点在寻找核心点集合中暂时被标记为噪声的点，但并不是最终的噪声点
    core_samples_mask[db.core_sample_indices_] = True
    print(core_samples_mask)
    labels = db.labels_
    #分类个数
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)


if __name__ == "__main__":
    show_dbscan()
# -*- encoding:utf-8 -*-
#无监督算法
from sklearn.neighbors import NearestNeighbors
import numpy as np 


x = np.array([[-1,1],[-2,-1],[-3,-2],[1,1],[2,1],[3,2]])
#训练过程
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(x)
distances, indices = nbrs.kneighbors(x)
print(distances)
print(indices)
print(nbrs)
#可视化过程
print(nbrs.kneighbors_graph(x).toarray())

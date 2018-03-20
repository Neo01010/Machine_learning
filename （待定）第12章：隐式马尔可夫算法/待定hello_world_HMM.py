import numpy as np
import matplotlib.pyplot as plt

from hmmlearn import hmm
#startprob:隐藏状态初始分布 transmat：状态转移矩阵 
startprob = np.array([0.6, 0.3, 0.1, 0.0])
transmat = np.array([[0.7, 0.2, 0.0, 0.1],
                     [0.3, 0.5, 0.2, 0.0],
                     [0.0, 0.3, 0.5, 0.2],
                     [0.2, 0.0, 0.2, 0.6]])
#means:代表各个隐藏状态对应的高斯分布期望向量s形成的矩阵        
means = np.array([[0.0,  0.0],
                  [0.0, 11.0],
                  [9.0, 10.0],
                  [11.0, -1.0]])

#每一个component之间的协方差
#identity（）:建立2维方形矩阵 tile:原矩阵的基础上进行翻倍
#covars表示各个隐藏状态对应的高斯分布协方差矩阵E形成的三维张量
covars = .5 * np.tile(np.identity(2), (4, 1, 1))
print(covars)
#hmm.GaussianHMM()假设观测状态符合高斯分布
model = hmm.GaussianHMM(n_components=4, covariance_type="full")

model.startprob_ = startprob
model.transmat_ = transmat
model.means_ = means
model.covars_ = covars

#生成样本
X, Z = model.sample(500)
#plt.plot绘制散点图
plt.plot(X[:, 0], X[:, 1], ".-", label = "observations", ms=6, mfc="orange", alpha=0.7)
for i, m in enumerate(means):
    plt.text(m[0], m[1], 'Component %i' % (i + 1),
             size=17, horizontalalignment='center',
             bbox=dict(alpha=.7, facecolor='w'))
plt.legend(loc='best')
plt.show()
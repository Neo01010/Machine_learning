# -*- encoding:utf-8 -*-
import numpy as np
#np.array()用于创建数组
a = np.array([1,2,3,4])
#dtype查看数据类型
print(a.dtype)
#shape查看数组大小
print(a.shape)

#使用arange创建array
a = np.arange(10)
print(type(a), str(a[2:5]))

#下标获取生成的数组是原始数组的视图，共享一块数据区
print(a)
b = a[3:6]
#b[1] = 0
print(a)

#matrix创建的是矩阵对象，采用矩阵运算
b = np.matrix([[1, 2, 3], [5, 5, 6], [7, 9, 9]])
print(type(b), b)
print(b*b**-1)
c = np.arange(10)
#对于一维数组，计算的是点集，如果要当做矢量进行矩阵运算，使用reshape  (-1,1):列向量  （1，-1）:行向量
print(a*c)
print(a*c.reshape((-1, 1)))
print(a*c.reshape((1, -1)))

#dot、inner、outter运算
#dot一维数组为内集，二维为矩阵乘积，多维为多个子矩阵的乘积
#inner?
#outter:只计算一维数组，如果是多维数组，先展为一维数组，再计算

#inv函数计算逆矩阵  solve求解多元一次方程组


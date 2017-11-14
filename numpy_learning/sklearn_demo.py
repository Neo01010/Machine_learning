# -*- encoding:utf-8 -*-


from sklearn import preprocessing
"""数字数组预处理"""
from sklearn.feature_extraction import DictVectorizer
"""文本型"""
from sklearn.feature_extraction.text import CountVectorizer
"""文本词袋特征提取"""
import numpy as np

"""
x = np.array([[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]])

#1：数组的标准化
x_scaled = preprocessing.scale(x)
print(x_scaled)

#2：数组正则化 norm:p-范数,用范数对数据进行稀疏化。
# 稀疏化使用能够实现特征的自动选择，以及模型的可解释性
#    不选择，复制的就是整行
x_normalized = preprocessing.normalize(x, norm='l2')
print(x_normalized)
x_normalized = preprocessing.normalize(x, norm='l1')
print(x_normalized)

#3：归一化
min_max_scaler = preprocessing.MinMaxScaler()
x_train_minmax = min_max_scaler.fit_transform(x)
print(x_train_minmax)

"""

#文本型特征提取
measurements = [
    {'city':'Dubai', 'temperature':33},
    {'city':'London', 'temperature':12},
    {'city':'San Fransisco', 'temperature':18},    
]
vec = DictVectorizer()
feature_array = vec.fit_transform(measurements).toarray()
print(feature_array)

#文本特征，词集、词袋
vectorizer = CountVectorizer()
print(vectorizer)
corpus = [
    'This is the first document',
    'this is the second second document.',
    'And the third one',
    'Is this the first document?',
]
X = vectorizer.fit_transform(corpus)
#print(X)
print(vectorizer.get_feature_names())
#对词袋的特征进行向量化
print(X.toarray())#词袋的特征空间叫做词汇表vocabulary：
#利用现有的词汇表对其他文本进行词袋处理？？？
vocabulary = vectorizer.vocabulary_
new_vectorizer = CountVectorizer(min_df=1, vocabulary = vocabulary)

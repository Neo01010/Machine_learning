# -*- encoding:utf-8 -*-

import tensorflow as tf
import numpy as np

training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename = "/home/ren/Desktop/machine_learning/numpy_learning/iris.csv",
    target_dtype = np.str,
    features_dtype = np.str
)

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
print(feature_columns)
x = training_set.data
y = training_set.target

print(x)
print(y)
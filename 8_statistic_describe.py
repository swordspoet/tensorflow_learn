import os
import tensorflow as tf
import numpy as np

# 限定TensorFlow输出的日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1. 向量的范数
# 2范数
a = tf.ones([2, 2])
b = tf.norm(a)
print(b.numpy())
# 1范数
# ord指定是求解几范数，默认是2范数；axis指定某个维度，0指行，1指列
c = tf.norm(a, ord=1, axis=1)
print(c.numpy())  # [2. 2.]

# 2. 聚合
# 2.1 返回值
# reduce_min/max/mean，既可以对全局又可以对某个维度（axis）计算

# 2.2 返回值所在的位置
# tf.argmax/argmin

# 3. 比较
# 比较两个tensor相同位置的值相同的个数
tensor_a = tf.constant([1, 2, 3])
tensor_b = tf.range(3)
res = tf.equal(tensor_a, tensor_b)  # <tf.Tensor: id=22, shape=(3,), dtype=bool, numpy=array([False, False, False])>

# 如何计算精确率？
np_array = np.array([[0.1, 0.2, 0.6, 0.1], [0.43, 0.17, 0.19, 0.11]])
tensor_1 = tf.convert_to_tensor(np_array, dtype=tf.float32)
prediction = tf.cast(tf.argmax(tensor_1, axis=1), dtype=tf.int32)
y = tf.convert_to_tensor([2, 1])
print(tf.equal(prediction, y))  # tf.Tensor([ True False], shape=(2,), dtype=bool)
correct = tf.reduce_sum(tf.cast(tf.equal(prediction, y), dtype=tf.int32))
accuracy = correct / 2  # <tf.Tensor: id=59, shape=(), dtype=float64, numpy=0.5>

# 4. unique去重

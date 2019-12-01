import os
import tensorflow as tf

# 限定TensorFlow输出的日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1. tf.sort/argsort
# 前者对张量直接排序，后者返回排序值在原tensor中的索引位置
tensor_a = tf.random.shuffle(tf.range(5))
print(tensor_a)
print(tf.sort(tensor_a, direction='DESCENDING'))  # tf.Tensor([4 3 2 1 0], shape=(5,), dtype=int32)
print(tf.argsort(tensor_a, direction='DESCENDING'))  # tf.Tensor([2 4 3 0 1], shape=(5,), dtype=int32)

# 2. top_k
# 返回张量前K个value以及索引
tensor_b = tf.random.uniform([3, 3], maxval=10, dtype=tf.int32)
top_k_res = tf.math.top_k(tensor_b, 2)
print(top_k_res.indices)  # 索引
print(top_k_res.values)   # 值
# 可以应用于求解top_k的精确值

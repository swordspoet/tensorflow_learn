import os
import tensorflow as tf

# 限定TensorFlow输出的日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1. tf.reshape
# 操作非常灵活，转换的时候需要考虑到reshape的物理意义
tensor_a = tf.random.normal([4, 28, 28, 3])
tf.reshape(tensor_a, [4, -1, 3])  # TensorShape([4, 784, 3])

# 2. tf.transpose
# 不改变图片本身的内容（content）
tf.transpose(tensor_a).shape                     # TensorShape([3, 28, 28, 4])
# 改变了图片本身的内容（content）
tf.transpose(tensor_a, perm=[0, 1, 3, 2]).shape  # TensorShape([4, 28, 3, 28])

# 3. 增加/减少维度
# tf.expand_dims()
# tf.squeeze()

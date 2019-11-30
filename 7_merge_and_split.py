import os
import tensorflow as tf

# 限定TensorFlow输出的日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 合并与切割操作不会影响维度的增加或者减少
# tf.concat
# tf.split

# 创建新的维度
# tf.stack
# tf.unstack

# tf.split相对于tf.unstack更加灵活

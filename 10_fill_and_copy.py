import os
import tensorflow as tf

# 限定TensorFlow输出的日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1. pad
# tf.pad(tensor, [0, 0], [0, 1])，
# [0, 0]表示对“行”维度的padding，[0, 1]表示对”列“维度的padding
# 上面的代码表示的是对张量的右边新增一列
# padding在图片和自然语言处理之中应用非常频繁

# 2. tile：数据复制操作（在内存中真实地复制数据）
# tf.tile(tensor, [1, 2]) 表示第一个维度保持不变，第二个维度复制一遍
tensor = tf.reshape(tf.range(9), [3, 3])
print(tf.tile(tensor, [2, 3]))  # 这里是先对第一个维度复制，然后再对第二个维度复制
# 相对于broadcast的效率要弱一点

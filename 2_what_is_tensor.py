import os
import tensorflow as tf

# 限定TensorFlow输出的日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1. 创建一个tensor常量，dtype可以制定数据类型，如float、double、int
print(tf.constant(1, dtype=tf.float32))

# 2. Tensor Property
# tensor_1是创建在CPU设备上的
tensor_1 = tf.constant("hello, world")
print(tensor_1.device)  # /job:localhost/replica:0/task:0/device:CPU:0

# tensor_2是创建在GPU设备上的，注意字符串是不能在GPU上面创建
with tf.device('gpu'):
    # tensor_2 = tf.constant("hello, world")
    tensor_2 = tf.range(10)
print(tensor_2.device)  # /job:localhost/replica:0/task:0/device:GPU:0

# 3. GPU上的tensor也可以“转移”到CPU上
print(tensor_2.cpu().device)

# 4. numpy数据转化为tensor：tf.convert_to_tensor()和tf.cast()

# 5. tf.Variable
# 需要被求导的参数需要经过tf.Variable包装一下，然后参数就具备了可以训练的属性

tensor_3 = tf.range(5)
b = tf.Variable(tensor_3)
print(b.dtype)      # 数据类型: <dtype: 'int32'>
print(b.trainable)  # 表明是可以被训练的: True

# 6. tensor转化为numpy
tensor_3.numpy()

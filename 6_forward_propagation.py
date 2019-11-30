import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets

# 限定TensorFlow输出的日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# x: [60K, 28, 28]
# y: [60K]
(x, y), _ = datasets.mnist.load_data()

x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.  # 归一化
y = tf.convert_to_tensor(y, dtype=tf.int32)

print(x.shape, y.shape, x.dtype, y.dtype)
print(tf.reduce_min(x), tf.reduce_max(x))
print(tf.reduce_min(y), tf.reduce_max(y))

# 创建一个数据集对象，方便一次取一个批次数据（128）
train_db = tf.data.Dataset.from_tensor_slices((x, y)).batch(128)
train_iter = iter(train_db)

# 创建权值: [b, 784] => [b, 256] => [b, 128] => [b, 10]
# 我们将需要优化的参数用tf.Variable()抱起来，这样TensorFlow才会在网络中对其求导
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

# 设置学习率
learning_rate = 1e-3

for epoch in range(10):
    # 对整个数据集迭代10次
    for step, (x, y) in enumerate(train_db):
        # 将输入“打平”: [b, 28*28] => [b, 256]
        x = tf.reshape(x, [-1, 28*28])

        # 将需要计算梯度的代码放入with中
        with tf.GradientTape() as tape:
            # [b, 784] => [b, 256]
            h1 = x@w1 + b1
            h1 = tf.nn.relu(h1)
            # [b, 256] => [b, 128]
            h2 = h1@w2 + b2
            h2 = tf.nn.relu(h2)
            # [b, 128] => [b, 10]
            out = h2@w3 + b3

            # 计算误差:均方差
            # 先对数据集做one_hot encoding
            y_oneHot = tf.one_hot(y, depth=10)
            loss = tf.reduce_mean(tf.square(y_oneHot - out))

        # 计算梯度： w = w - lr * w_gradient
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        # 右边的w1是tf.Variable()类型，然而右边的w1已经转化为了tf.tensor类型，
        # 所以，继续对右边的w1优化会出错，这里需要针对w1做原地更新的操作w1.assign_sub，
        # 它可以保持数据的类型不变
        # 可以通过print(isinstance(w1, tf.Variable))检验w1的数据类型
        w1.assign_sub(learning_rate * grads[0])
        b1.assign_sub(learning_rate * grads[1])
        w2.assign_sub(learning_rate * grads[2])
        b2.assign_sub(learning_rate * grads[3])
        w3.assign_sub(learning_rate * grads[4])
        b3.assign_sub(learning_rate * grads[5])

        # 每100个step打印一次损失
        if step % 100 == 0:
            print(epoch, step, 'loss', float(loss))

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers


# 设置日志打印的级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# MNIST数据集：每个类别有7000张图片
# 从官方下载数据集，pycharm设置了代理之后需要重启才能生效
(x_train, y_train), _ = datasets.mnist.load_data()
print('datasets:', x_train.shape, y_train.shape)

# 数据集转换为tensor
x_train = tf.convert_to_tensor(x_train, dtype=tf.float32) / 255.
y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
y_train = tf.one_hot(y_train, depth=10)
# 转化为批次数据，一次针对多个数据做处理
db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = db.batch(200)

# 定义网络的工具
model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(256, activation="relu"),
    layers.Dense(10)])
# 优化器类
optimizer = optimizers.SGD(learning_rate=0.001)


def train_epoch(epoch):
    # 前向运算
    for step, (x, y) in enumerate(train_dataset):
        # TensorFlow的自动求导工具
        with tf.GradientTape() as tape:
            # x的维度为[batch, 28, 28]，即一个批次的数据，进入前向网络之前需要“打平”
            x = tf.reshape(x, (-1, 28*28))
            out = model(x)
            loss = tf.reduce_sum(tf.square(out - y)) / x.shape[0]

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print(epoch, step, loss.numpy())


def train():
    # 数据集大小为60K，30个epoch，一个批次200个图片，那么每个epoch会有300（60000/200）个step
    for epoch in range(30):
        train_epoch(epoch)


if __name__ == '__main__':
    train()

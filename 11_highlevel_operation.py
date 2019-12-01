import os
import tensorflow as tf
import matplotlib.pyplot as plt

# 限定TensorFlow输出的日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 1. tf.scatter_nd()
# 可以根据给定的索引精确定向更新该索引上的值

# 2. tf.where()
# 根据一个布尔值的数组对张量取数据

# 3. meshgrid

def func(x):
    """

    :param x: [b, 2]
    :return:
    """
    z = tf.math.sin(x[..., 0]) + tf.math.sin(x[..., 1])

    return z


y = tf.linspace(-2., 2, 5)
x = tf.linspace(-2., 2, 5)
points_x, points_y = tf.meshgrid(x, y)
points = tf.stack([points_x, points_y], axis=2)

z = func(points)
print('z:', z.shape)

plt.figure('plot 2d func value')
plt.imshow(z, origin='lower', interpolation='none')
plt.colorbar()

plt.figure('plot 2d func contour')
plt.contour(points_x, points_y, z)
plt.colorbar()
plt.show()

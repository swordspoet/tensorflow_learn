import os
import tensorflow as tf

# 限定TensorFlow输出的日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1. 索引
# 不要用tensor[][]的索引方式，推荐numpy风格(Numpy-style indexing)的索引
# 例如，取tensor_a表示的是4张28*28拥有RGB三通道的图片，
# 那么取`第二张`图片的`第三行``第四列`的RGB元素的方式如下：
tensor_a = tf.random.normal([6, 28, 28, 3])
idx_numpy = tensor_a[1, 2, 3]

# 2. 切片
# [start:end)       “左闭右开”，start默认为0，end默认为-1
# [:]               start和end都不写则取该维度下所有数据
# [start:end:step]  step步长，表示从start到end区间，每隔step取一次数据
# [start:end:-1]    step为负数表示从右往左切片
tensor_b = tf.range(10)
print(tensor_b[2::-1])  # 第三个元素开始（包括）往左取数据，返回[2, 1, 0]
# ...               具备自动推导功能

# 3. 选择性索引（使用的地方比较少）
# tf.gather         根据自定义索引规则搜集数据，比之前的索引更加灵活
# 取第三、第四和第六张照片的数据
tf.gather(tensor_a, axis=0, indice=[2, 3, 5])
# tf.gather_nd      根据维度取数据
tf.gather_nd(tensor_a, [1, 2])            # 表示取第二张图片的第三行下所有的数据
tf.gather_nd(tensor_a, [[2, 3], [1, 2]])  # 表示取第二张图片的第四行和第二张图片的第三行数据

# 4. tf.boolean_mask
# 取或者不取某个维度下的数据

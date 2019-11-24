import os
import tensorflow as tf

# 限定TensorFlow输出的日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# broadcasting默认从最小（最右）的维度开始扩展
# 优势：减少内存占用空间
# 扩展方式：显式扩展和隐式扩展

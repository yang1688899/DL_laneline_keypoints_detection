import tensorflow as tf
import utils
import config
from tensorflow.contrib.layers import flatten
from tensorflow.keras.layers import LeakyReLU,Dense,GlobalAveragePooling3D

def weight_variable(shape, stddev=0.1, name=None):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial,name=name)


def bias_variable(shape, bais=0.1, name=None):
    initial = tf.constant(bais, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, w, name=None):
    return tf.nn.conv2d(x, w, [1, 1, 1, 1], 'SAME', name=name)


def max_pool_2x2(x, name=None):
    return tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name=name)


def max_pool_3x3(x, name=None):
    return tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME', name=name)


def avg_pool_3x3(x, name=None):
    return tf.nn.avg_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME', name=name)

def net(inteception_out,rate):
    with tf.name_scope("bottom"):
        global_avg = GlobalAveragePooling3D(inteception_out)

        fc1_w = weight_variable([2048,1024], name="my_fc1_w")
        fc1_b = weight_variable([1024], name="my_fc1_b")
        fc1 = tf.nn.relu(tf.matmul(global_avg,fc1_w)+fc1_b, name="my_fc1")
        drop_fc1 = tf.nn.dropout(fc1,keep_prob=rate)

        fc3_w = weight_variable([1024,512], name="my_fc3_w")
        fc3_b = bias_variable([512], name="my_fc3_b")
        fc3 = tf.nn.relu(tf.matmul(drop_fc1,fc3_w)+fc3_b, name="my_fc3")
        drop_fc3 = tf.nn.dropout(fc3,keep_prob=rate)

        fc4_w = weight_variable([512,27], name="my_fc4_w")
        fc4_b = bias_variable([27], name="my_fc4_b")
        prediction = tf.add(tf.matmul(drop_fc3,fc4_w),fc4_b, name="prediction")

    return prediction



class Network:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 480, 640, 3], name="input_x")
        self.y = tf.placeholder(dtype=tf.float32,shape=[None,27])
        self.rate = tf.placeholder(dtype=tf.float32, name="rate")
        inception = tf.keras.applications.InceptionV3(include_top=False, input_tensor=self.x)
        self.inception_out = inception.output
        self.prediction = net(self.inception_out,self.rate)






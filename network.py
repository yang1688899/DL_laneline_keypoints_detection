import tensorflow as tf
import utils
import config
from tensorflow.contrib.layers import flatten

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

def net(x,rate):
    conv1_w = weight_variable([15,20,512,32],name="my_conv1_w")
    conv1_b = bias_variable([32], name="my_conv1_b")
    conv1 = tf.nn.relu(conv2d(x,conv1_w)+conv1_b, name="my_conv1")

    flat = flatten(conv1)
    drop_flat = tf.nn.dropout(flat,keep_prob=rate)

    fc2_w = weight_variable([9600,1024], name="my_fc2_w")
    fc2_b = weight_variable([1024], name="my_fc2_b")
    fc2 = tf.nn.relu(tf.matmul(drop_flat,fc2_w)+fc2_b, name="my_fc2")
    drop_fc2 = tf.nn.dropout(fc2,keep_prob=rate)

    fc3_w = weight_variable([1024,512], name="my_fc3_w")
    fc3_b = bias_variable([512], name="my_fc3_b")
    fc3 = tf.nn.relu(tf.matmul(drop_fc2,fc3_w)+fc3_b, name="my_fc3")
    drop_fc3 = tf.nn.dropout(fc3,keep_prob=rate)

    fc4_w = weight_variable([512,27], name="my_fc4_w")
    fc4_b = bias_variable([27], name="my_fc4_b")
    prediction = tf.add(tf.matmul(drop_fc3,fc4_w),fc4_b, name="prediction")

    return prediction

class Network:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 15, 20, 512], name="input_f")
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 27], name="input_y")
        self.rate = tf.placeholder(dtype=tf.float32, name="rate")
        self.prediction = net(self.x,self.rate)

        self.loss = tf.reduce_mean(tf.abs(self.prediction - self.y))
        tf.summary.scalar("loss",self.loss)




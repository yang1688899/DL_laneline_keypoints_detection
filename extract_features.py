from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import glob
import config
import numpy as np
import utils
import tensorflow as tf
import network
import os

img_paths = np.array(glob.glob(config.DATADIR + '/train/*/*/*.jpg'))
label_paths = np.array(glob.glob(config.DATADIR + '/train/*/*/*/*.txt'))

img_paths,label_paths = shuffle(img_paths,label_paths)

#由于训练数据较少，使用cross_validation可以减小过拟合
kf = KFold(n_splits=4)

i=0
net = network.Network()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for train_index,valid_index in kf.split(img_paths):
        train_img_paths = img_paths[train_index]
        train_label_paths = label_paths[train_index]
        valid_img_paths = img_paths[valid_index]
        valid_label_paths = label_paths[valid_index]

        record_dir = "./record_%s"%i
        if not os.path.exists(record_dir):
            os.mkdir(record_dir)
        utils.convert_to_tfrecords(sess,net,train_img_paths,train_label_paths,
                                   savefile=record_dir+ "/train.tfrecords")
        utils.convert_to_tfrecords(sess, net, valid_img_paths, valid_label_paths,
                                   savefile=record_dir + "/valid.tfrecords")
        i+=1

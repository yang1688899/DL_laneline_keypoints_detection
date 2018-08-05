import network
import utils
import config
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split
import glob
from sklearn.utils import shuffle
import os

rate = tf.placeholder(dtype=tf.float32,name="rate")
train_feature, train_label = utils.get_from_tfrecords(["./record_0/train.tfrecords"])
train_features, train_labels = utils.get_batch(train_feature, train_label, config.BATCH_SIZE, 800)
prediction = network.net(train_features,rate)
train_loss = tf.reduce_mean(tf.square(prediction - train_labels))

valid_feature, valid_label = utils.get_from_tfrecords(["./record_0/valid.tfrecords"])
valid_features,valid_labels = utils.get_batch(valid_feature,valid_label,config.BATCH_SIZE,800,is_shuffle=False)
valid_prediction = network.net(valid_features,rate)
valid_loss = tf.reduce_mean(tf.square(valid_prediction - valid_labels))

train_step = tf.train.AdamOptimizer().minimize(train_loss)

saver = tf.train.Saver()
logger = utils.get_logger("./log/info.log")

with tf.Session() as sess:
    sess, step = utils.start_or_restore_training(sess, saver, checkpoint_dir=config.CKECKDIR)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    start_time = time.time()
    print("trainning......")
    while True:
        sess.run(train_step,feed_dict={rate:0.5})

        step += 1

        if step%50==0:
            train_loss_v = sess.run(train_loss,feed_dict={rate:1.})
            valid_loss_v = utils.validation(sess, valid_loss,rate, batch_size=config.BATCH_SIZE)
            duration = time.time() - start_time
            logger.info("step %d: trainning loss is %g, validation loss is %g (%0.3f sec)" % (step, train_loss_v,valid_loss_v, duration))
            print("step %d: trainning loss is %g, validation loss is %g (%0.3f sec)" % (step, train_loss_v,valid_loss_v, duration))
            start_time = time.time()
        if step%1000==0:
            if not os.path.exists(config.CKECKDIR):
                os.mkdir(config.CKECKDIR)
            saver.save(sess, config.CKECKFILE, global_step=step)
            print('writing checkpoint at step %s' % step)




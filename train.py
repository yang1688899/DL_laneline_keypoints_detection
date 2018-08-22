import network
import utils
import config
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split
import glob
from sklearn.utils import shuffle
import os
import numpy as np

img_paths = np.array(glob.glob(config.DATADIR + '/train/*/*/*.jpg'))
label_paths = np.array(glob.glob(config.DATADIR + '/train/*/*/*/*.txt'))

img_paths,label_paths = shuffle(img_paths,label_paths)

net = network.Network()


train_loss = tf.reduce_mean(tf.square(net.prediction - net.y))
bottom_var = tf.get_collection(tf.GraphKyes.TRAINABLE_VARIABLES, scope='bottom')
train_step_freeze_top = tf.train.AdamOptimizer(1e-3).minimize(train_loss,var_list=bottom_var)
train_step_all = tf.train.AdamOptimizer(1e-4).minimize(train_loss)



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




import network
import utils
import config
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split
import glob
from sklearn.utils import shuffle

train_feature, train_label = utils.get_from_tfrecords(["./record_0/train.tfrecords"])
train_features, train_labels = utils.get_batch(train_feature, train_label, config.BATCH_SIZE, 800)

valid_feature, valid_label = utils.get_from_tfrecords(["./record_0/valid.tfrecords"])
valid_features, valid_labels = utils.get_batch(valid_feature, valid_label, config.BATCH_SIZE, 800, is_shuffle=False)

net = network.Network()

train_step = tf.train.AdamOptimizer().minimize(net.loss)

saver = tf.train.Saver()
logger = utils.get_logger("./log/info.log")

with tf.Session() as sess:
    sess, step = utils.start_or_restore_training(sess, saver, checkpoint_dir=config.CKECKDIR)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    summary_all = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter("./summary", graph=sess.graph)
    start_time = time.time()
    print("trainning......")

    while True:
        train_features_batch,train_labels_batch = sess.run([train_features, train_labels])
        sess.run(train_step,feed_dict={net.x:train_features_batch, net.y:train_labels_batch, net.rate:0.5})

        step += 1

        if step%100==0:
            train_loss,summary_info = sess.run([net.loss,summary_all],feed_dict={net.x:train_features_batch, net.y:train_labels_batch, net.rate:1.})
            summary_writer.add_summary(summary_info,global_step=step)
            valid_loss = utils.validation(sess, net, valid_features, valid_labels, batch_size=config.BATCH_SIZE)
            duration = time.time() - start_time
            logger.info("step %d: trainning loss is %g, validation loss is %g (%0.3f sec)" % (step, train_loss,valid_loss, duration))
            print("step %d: trainning loss is %g, validation loss is %g (%0.3f sec)" % (step, train_loss,valid_loss, duration))
            start_time = time.time()
        if step%1000==0:
            saver.save(sess,'./checkpoints/model.ckpt', global_step=step)
            print('writing checkpoint at step %s' % step)



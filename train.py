import network
import utils
import config
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split
import glob
from sklearn.utils import shuffle

img_paths = glob.glob(config.DATADIR + '/train/*/*/*.jpg')
label_paths = glob.glob(config.DATADIR + '/train/*/*/*/*.txt')

img_paths,label_paths = shuffle(img_paths,label_paths)
train_img_paths,valid_img_paths,train_label_paths,valid_label_paths = train_test_split(img_paths,label_paths,test_size=0.2)

net = network.Network()

train_step = tf.train.AdamOptimizer().minimize(net.loss)

saver = tf.train.Saver()
data_gen = utils.data_generator(train_img_paths,train_label_paths,batch_size=config.BATCH_SIZE,is_shuffle=True)
logger = utils.get_logger("./log/info.log")

with tf.Session() as sess:
    sess, step = utils.start_or_restore_training(sess, saver, checkpoint_dir=config.CKECKDIR)
    start_time = time.time()
    print("trainning......")
    while True:
        features,labels = next(data_gen)

        extract_features = sess.run(net.vgg_no_top, feed_dict={net.x:features})

        sess.run(train_step,feed_dict={net.f:extract_features, net.y:labels, net.rate:0.5})

        step += 1

        if step%100==0:
            train_loss = sess.run(net.loss,feed_dict={net.x:extract_features, net.y:labels, net.rate:1.})
            valid_loss = utils.validation(sess, net, valid_img_paths, valid_label_paths, batch_size=config.BATCH_SIZE)
            duration = time.time() - start_time
            logger.info("step %d: trainning loss is %g, validation loss is %g (%0.3f sec)" % (step, train_loss,valid_loss, duration))
            start_time = time.time()
        if step%1000==0:
            saver.save(sess, config.CHECKFILE, global_step=step)
            print('writing checkpoint at step %s' % step)




import glob
import config
import cv2
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
import os
import logging
from math import ceil

#含中文路径读取图片方法
def cv_imread(filepath):
    img = cv2.imdecode(np.fromfile(filepath,dtype=np.uint8),-1)
    return img

def get_logger(filepath,level=logging.INFO):
    dir = os.path.dirname(filepath)
    if not os.path.exists(dir):
        os.mkdir(dir)
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    # create a file handler
    handler = logging.FileHandler(filepath)
    handler.setLevel(logging.INFO)

    # create a logging format
    #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)
    return logger

#初始化sess,或回复保存的sess
def start_or_restore_training(sess,saver,checkpoint_dir):
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print('Restore the model from checkpoint %s' % ckpt.model_checkpoint_path)
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
    else:
        sess.run(tf.global_variables_initializer())
        step = 1
        print('start training from new state')
    return sess,step

#把数据打包，转换成tfrecords格式，以便后续高效读取
def convert_to_tfrecords(sess,net,img_paths,labels_paths,savefile):
    writer = tf.python_io.TFRecordWriter(savefile)
    num_example = len(img_paths)
    data_gen = data_generator(img_paths,labels_paths,batch_size=config.BATCH_SIZE,is_shuffle=False)
    num_it = ceil(num_example/config.BATCH_SIZE)
    writed_num = 0
    for i in range(num_it):
        features,labels = next(data_gen)
        #提取特征
        extract_features = sess.run(net.vgg_no_top,feed_dict={net.x:features})
        for f,l in zip(extract_features,labels):
            height, width, depth = f.shape

            example = tf.train.Example(features=tf.train.Features(feature={
                'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[depth])),
                'feature': tf.train.Feature(bytes_list=tf.train.BytesList(value=[f.tobytes()])),
                'label': tf.train.Feature(float_list=tf.train.FloatList(value=l))
            }))

            serialized = example.SerializeToString()
            writer.write(serialized)
            writed_num += 1

    print("pass %s sample, convert %s sample"%(num_example,writed_num))
    writer.close()

#从tfrecords中解压获取图片
def get_from_tfrecords(filepaths,num_epoch=None):
    filename_queue = tf.train.string_input_producer(filepaths,num_epochs=num_epoch)  # 因为有的训练数据过于庞大，被分成了很多个文件，所以第一个参数就是文件列表名参数
    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)
    example = tf.parse_single_example(serialized, features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'depth': tf.FixedLenFeature([], tf.int64),
        'feature': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.float64)
    })
    label = tf.cast(example['label'], tf.float32)
    img = tf.decode_raw(example['img'], tf.uint8)
    img = tf.reshape(img, [
        tf.cast(example['height'], tf.int32),
        tf.cast(example['width'], tf.int32),
        tf.cast(example['depth'], tf.int32)])

    # label=example['label']
    return img, label

def extract_keypoints(filepath):
    with open(filepath, 'r') as file:
        data = file.readlines()
        if len(data)==29:
            points_info = data[3:21]
        else:
            points_info = data[2:20]
        keypoints = [point.replace('\t', "").replace('\n', "").split(',')[1:3] for point in points_info]
    return keypoints

def convert_keypoints_to_label(keypoints):
    y = []
    left_x = []
    right_x = []
    for i, point in enumerate(keypoints):
        y_cord = float(point[1])
        if not y_cord in y:
            y.append(y_cord)
        if i % 2 == 0:
            left_x.append(float(point[0]))
        else:
            right_x.append(float(point[0]))
    y.extend(left_x)
    y.extend(right_x)
    return np.array(y)

def get_label(filepath):
    keypoints = extract_keypoints(filepath)
    label = convert_keypoints_to_label(keypoints)
    return label

def get_feature(imgpath):
    img = cv_imread(imgpath)
    return (img-128.)/255.

def data_generator(img_paths,label_paths,batch_size=64,is_shuffle=True):
    num_sample = len(img_paths)
    if is_shuffle:
        img_paths,label_paths = shuffle(img_paths,label_paths)
    while True:
        for offset in range(0,num_sample,batch_size):
            if is_shuffle:
                img_paths, label_paths = shuffle(img_paths, label_paths)
            batch_img_paths = img_paths[offset : offset+batch_size]
            batch_label_paths = label_paths[offset : offset+batch_size]
            features = []
            labels = []
            for imgpath,labelpath in zip(batch_img_paths,batch_label_paths):
                features.append(get_feature(imgpath))
                labels.append(get_label(labelpath))
            yield np.array(features), np.array(labels)

def validation(sess,net,valid_tf_path,batch_size=64):
    valid_feature, valid_label = get_from_tfrecords(["./record_0/valid.tfrecords"])
    valid_features, valid_labels = get_batch(valid_feature, valid_label, config.BATCH_SIZE, 800, is_shuffle=False)
    num_it = ceil( 800/batch_size )
    total_loss = 0
    for i in range(num_it):
        valid_features_batch,valid_labels_batch = sess.run([valid_features, valid_labels])
        total_loss += sess.run(net.loss, feed_dict={net.x: valid_features_batch, net.y: valid_labels_batch, net.rate: 1.0})
    return total_loss/num_it



# generator = data_generator(batch_size=1)
# for i in range(1000):
#     features,labels = next(generator)
#     print(features.shape)
#     print(labels.shape)
#     print(labels)

# label_paths = glob.glob(config.DATADIR + "/train/*/*/*/*.txt")
# # extract_keypoints(label_paths[0])
# for i in range(100):
#     print(label_paths[i])
#     get_label(label_paths[i])







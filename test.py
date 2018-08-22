import utils
import network
import tensorflow as tf
import config
import glob
import cv2
import os
import numpy as np


net = network.Network()
saver = tf.train.Saver()

test_paths = glob.glob(config.DATADIR+"/test/*/*/*/*.jpg")

with (tf.Session()) as sess:
    sess,_ = utils.start_or_restore_training(sess,saver,config.CKECKDIR)

    for path in test_paths:
        origin_img = utils.cv_imread(path)
        img = utils.get_feature(path)
        prediction = utils.make_predict(sess, net, img)
        lines = utils.prediction2cords(prediction)

        draw_img = utils.draw_lines(origin_img,lines)
        # cv2.polylines(origin_img, left_line, isClosed=False,color=(0,0,255),thickness=5,lineType=8)

        cv2.imshow("temp",draw_img)
        cv2.waitKey()

        # points = np.array([[910, 641], [206, 632], [696, 488], [458, 485]])
        # # points.dtype => 'int64'
        # cv2.polylines(origin_img, np.int32([points]), 1, (255, 255, 255))



import glob
import config
import cv2
import numpy as np
from sklearn.utils import shuffle

#含中文路径读取图片方法
def cv_imread(filepath):
    img = cv2.imdecode(np.fromfile(filepath,dtype=np.uint8),-1)
    return img

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
        y_cord = float(point[1])/config.WIDTH
        if not y_cord in y:
            y.append(y_cord)
        if i % 2 == 0:
            left_x.append(float(point[0])/config.HEIGHT)
        else:
            right_x.append(float(point[0])/config.HEIGHT)
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

def data_gen(batch_size=128):
    img_paths = glob.glob(config.DATADIR + "/train/*/*/*.jpg")
    label_paths = glob.glob(config.DATADIR + "/train/*/*/*/*.txt")
    num_sample = len(img_paths)
    img_paths,label_paths = shuffle(img_paths,label_paths)
    while True:
        for offset in range(0,batch_size,num_sample):
            img_paths, label_paths = shuffle(img_paths, label_paths)
            batch_img_paths = img_paths[offset : offset+batch_size]
            batch_label_paths = label_paths[offset : offset+batch_size]
            features = []
            labels = []
            for imgpath,labelpath in zip(batch_img_paths,batch_label_paths):
                features.append(get_feature(imgpath))
                labels.append(get_label(labelpath))
            yield np.array(features), np.array(labels)






generator = data_gen(batch_size=32)
for i in range(1000):
    features,labels = next(generator)
    print(features.shape)
    print(labels.shape)

# label_paths = glob.glob(config.DATADIR + "/train/*/*/*/*.txt")
# # extract_keypoints(label_paths[0])
# for i in range(100):
#     print(label_paths[i])
#     get_label(label_paths[i])







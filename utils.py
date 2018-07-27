import glob
import config
import cv2
import numpy as np
import csv

#含中文路径读取图片方法
def cv_imread(filepath):
    img = cv2.imdecode(np.fromfile(filepath,dtype=np.uint8),-1)
    return img

def get_labels():
    label_paths = glob.glob(config.DATADIR + "/train/*/*/*/*.txt")

    for path in label_paths:
        with open(path, 'r') as file:
            data = file.readlines()
            points = data[3:21]

            label = [point.replace('\t', "").replace('\n', "").split(',')[1:3] for point in points]

            # label[]
            label = np.array(label).reshape((1, 9,-1))
            set(label[:][:][1])



# img_paths = glob.glob(config.DATADIR+"/train/*/*/*.jpg")
#
# for i in range(10000):
#     path = img_paths[i]
#     img = cv_imread(img_paths[0])
#     cv2.imshow("temp",img)
#     cv2.waitKey()
# cv2.imshow()
#
# print(img_paths)
label_paths = glob.glob(config.DATADIR + "/train/*/*/*/*.txt")
with open(label_paths[0], 'r') as file:
    data = file.readlines()
    points = data[3:21]

    label = [point.replace('\t', "").replace('\n', "").split(',')[1:3] for point in points]


    y = []
    left_x = []
    right_x = []
    for i,point in enumerate(label):
        if not float(point[1]) in y:
            y.append(float(point[1]))
        if i%2 == 0:
            left_x.append(float(point[0]))
        else:
            right_x.append(float(point[0]))

    print (y)
    print(left_x)
    print(right_x)






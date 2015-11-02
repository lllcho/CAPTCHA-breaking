# __author__ = 'lllcho'
# __date__ = '2015/7/31'
# coding=utf8
import os
import glob
import numpy as np
import cv2
import codecs
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D

np.random.seed(123)
nb_class = 36
letters = list('0123456789abcdefghijklmnopqrstuvwxyz')
model_path = './model/type2_model.d5'

model = Sequential()
model.add(Convolution2D(32, 1, 4, 4, border_mode='full', activation='relu'))
model.add(Convolution2D(32, 32, 4, 4, activation='relu'))
model.add(MaxPooling2D(poolsize=(3, 3)))
model.add(Dropout(0.25))
model.add(Convolution2D(64, 32, 4, 4, border_mode='full', activation='relu'))
model.add(Convolution2D(64, 64, 4, 4, activation='relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64 * 5 * 5, 512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, nb_class, activation='softmax'))
model.load_weights(model_path)
model.compile(loss='categorical_crossentropy', optimizer='adagrad')

comp = 'type2_test1'
img_dir = './image/' + comp + '/'
fcsv = codecs.open("./result/" + comp + '.csv', 'w', 'utf-8')
# for nb_img in range(1,20001):
#     name=comp+ '_' + str(nb_img) + '.jpg'
import os
names = os.listdir(img_dir)
for name in names:
    imgname = img_dir + name
    print imgname
    img = cv2.imread(imgname, cv2.IMREAD_GRAYSCALE)
    I0 = cv2.morphologyEx(255 - img, cv2.MORPH_CLOSE, np.ones((5, 5), dtype='uint8'))
    I1 = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, np.ones((5, 5), dtype='uint8'))
    img_closed = cv2.add(I0, I1)

    retval, t = cv2.threshold(img_closed, 125, 1, cv2.THRESH_BINARY)
    h_sum = t.sum(axis=0)
    v_sum = t.sum(axis=1)
    x1, x2 = (v_sum > 1).nonzero()[0][0], (v_sum > 1).nonzero()[0][-1]
    y1, y2 = (h_sum > 5).nonzero()[0][0], (h_sum > 1).nonzero()[0][-1]
    im = img[x1:x2, y1:y2]

    imgs = np.zeros((5, 1, 32, 32), dtype=np.uint8)
    t = im.shape[1] / 5.0
    dd = 4
    bb = np.zeros((im.shape[0], dd), dtype=np.uint8) + 255
    im1 = im.transpose()[0:np.floor(t) + dd].transpose()
    im2 = im.transpose()[np.floor(t) - dd:np.floor(2 * t) + dd].transpose()
    im3 = im.transpose()[np.floor(2 * t) - dd:np.floor(3 * t) + dd].transpose()
    im4 = im.transpose()[np.floor(3 * t) - dd:np.floor(4 * t) + dd].transpose()
    im5 = im.transpose()[np.floor(4 * t) - dd:].transpose()
    imgs[0, 0] = cv2.resize(np.concatenate((bb, im1), axis=1), (32, 32))
    imgs[1, 0] = cv2.resize(im2, (32, 32))
    imgs[2, 0] = cv2.resize(im3, (32, 32))
    imgs[3, 0] = cv2.resize(im4, (32, 32))
    imgs[4, 0] = cv2.resize(np.concatenate((im5, bb), axis=1), (32, 32))

    imgs = imgs.astype('float32') / 255.0
    classes = model.predict_classes(imgs, verbose=0)
    result = []
    for c in classes:
        result.append(letters[c])
    print(''.join(result).upper())
    fcsv.write(name + ',' + ''.join(result).upper() + '\n')
fcsv.close()

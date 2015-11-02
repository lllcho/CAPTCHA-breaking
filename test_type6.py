# -*- coding: utf-8 -*-
# __author__ = 'lllcho'
# __date__ = '2015/10/2'
import cv2
import codecs
import cPickle
import numpy as np
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D

chars = cPickle.load(open('model/chars_type6.pkl', 'rb'))
nb_model = Sequential()
nb_model.add(Convolution2D(32, 1, 4, 4, border_mode='full', activation='relu'))
nb_model.add(Convolution2D(32, 32, 4, 4, activation='relu'))
nb_model.add(MaxPooling2D(poolsize=(3, 3)))
nb_model.add(Dropout(0.3))
nb_model.add(Convolution2D(64, 32, 4, 4, border_mode='full', activation='relu'))
nb_model.add(Convolution2D(64, 64, 4, 4, activation='relu'))
nb_model.add(MaxPooling2D(poolsize=(2, 3)))
nb_model.add(Dropout(0.3))
nb_model.add(Flatten())
nb_model.add(Dense(64 * 6 * 15, 512, activation='relu', ))
nb_model.add(Dropout(0.6))
nb_model.add(Dense(512, 3, activation='softmax'))
nb_model.load_weights('model/type6_nb_model.d5')
nb_model.compile(loss='categorical_crossentropy', optimizer='adagrad')

weight_decay = 0.001
chars_model = Sequential()
chars_model.add(Convolution2D(32, 1, 4, 4, border_mode='valid', activation='relu', W_regularizer=l2(weight_decay)))
chars_model.add(Convolution2D(32, 32, 4, 4, activation='relu', W_regularizer=l2(weight_decay)))
chars_model.add(MaxPooling2D(poolsize=(3, 3)))
chars_model.add(Dropout(0.3))
chars_model.add(Convolution2D(64, 32, 4, 4, border_mode='full', activation='relu', W_regularizer=l2(weight_decay)))
chars_model.add(Convolution2D(64, 64, 4, 4, activation='relu', W_regularizer=l2(weight_decay)))
chars_model.add(MaxPooling2D(poolsize=(2, 2)))
chars_model.add(Dropout(0.3))
chars_model.add(Flatten())
chars_model.add(Dense(64 * 5 * 4, 512, activation='relu', W_regularizer=l2(weight_decay)))
chars_model.add(Dropout(0.6))
chars_model.add(Dense(512, 993, activation='softmax'))
chars_model.load_weights('model/type6_chars_model.d5')
chars_model.compile(loss='categorical_crossentropy', optimizer='adagrad')

comp = 'type6_test1'
img_dir = './image/' + comp + '/'
f_csv = codecs.open("result/" + comp + '.csv', 'w', 'utf-8')
# for nb_img in range(1,20001):
#     name=comp+'_'+str(nb_img)+'.png'
import os

names = os.listdir(img_dir)
for name in names:
    imgname = img_dir + name
    img = cv2.imread(imgname, cv2.IMREAD_GRAYSCALE)
    retval, t = cv2.threshold(img, 200, 1, cv2.THRESH_BINARY_INV)
    s = t.sum(axis=0)
    y1, y2 = (s > np.median(s) + 5).nonzero()[0][0], (s > np.median(s) + 5).nonzero()[0][-1]
    im = img[8:-4, max(y1 - 5, 0):min(y2 + 5, img.shape[1])]
    im = 255 - im
    im0 = cv2.resize(im, (int(im.shape[1] * 0.75), 39))
    if im.shape[1] < 180:
        im = np.concatenate((im, np.zeros((52, 180 - im.shape[1]), dtype='uint8')), axis=1)
    else:
        im = cv2.resize(im, (180, 52))
    im = cv2.resize(im, (135, 39))
    # cv2.imshow('a',im0)
    # cv2.waitKey()
    n = nb_model.predict_classes(im.astype(np.float32).reshape((1, 1, 39, 135)) / 255, verbose=0) + 4
    start = 15
    im1 = np.zeros((39, 250), dtype=np.uint8)
    im1[:, start:im0.shape[1] + start] = im0
    step = im0.shape[1] / float(n)
    center = [i + step / 2 for i in np.arange(0, im0.shape[1], step).tolist()]
    imgs = np.zeros((n, 1, 39, start * 2), dtype=np.float32)
    for kk, c in enumerate(center):
        imgs[kk, 0, :, :] = im1[:, c:c + start * 2]
        # cv2.imshow('a',imgs[kk,0])
        # cv2.waitKey()
    classes = chars_model.predict_classes(imgs.astype(np.float32) / 255, verbose=0)
    result = []
    for c in classes:
        result.append(chars[c])
    print(name + ',' + ''.join(result).upper())
    f_csv.write(name + ',' + ''.join(result).upper() + '\n')
f_csv.close()

# __author__ = 'lllcho'
# __date__ = '2015/8/5'
import cv2
import numpy as np
import cPickle
import codecs
import scipy.spatial.distance
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from util import *

model_path = './model/chars4800_model.d5'
dic = cPickle.load(open('model/chars4800.pkl', 'rb'))
chars = dic['c']
pys = dic['p']
py_ch = dic['pc']
ch_py = dic['cp']
nb_class = 4800
model_ch = Sequential()
model_ch.add(Convolution2D(32, 1, 4, 4, border_mode='full', activation='relu'))
model_ch.add(Convolution2D(32, 32, 4, 4, activation='relu'))
model_ch.add(MaxPooling2D(poolsize=(2, 2)))
model_ch.add(Dropout(0.25))
model_ch.add(Convolution2D(64, 32, 4, 4, border_mode='full', activation='relu'))
model_ch.add(Convolution2D(64, 64, 4, 4, activation='relu'))
model_ch.add(MaxPooling2D(poolsize=(2, 2)))
model_ch.add(Dropout(0.25))
model_ch.add(Flatten())
model_ch.add(Dense(64 * 8 * 8, 512, activation='relu'))
model_ch.add(Dropout(0.5))
model_ch.add(Dense(512, nb_class, activation='softmax'))
model_ch.load_weights(model_path)
model_ch.compile(loss='categorical_crossentropy', optimizer='adagrad')

nb_class = 408
weight_decay = 0.001
model_py = Sequential()
model_py.add(Convolution2D(32, 1, 4, 4, border_mode='full', activation='relu', W_regularizer=l2(weight_decay)))
model_py.add(Convolution2D(32, 32, 4, 4, activation='relu', W_regularizer=l2(weight_decay)))
model_py.add(MaxPooling2D(poolsize=(2, 3)))
model_py.add(Dropout(0.25))
model_py.add(Convolution2D(64, 32, 4, 4, border_mode='full', activation='relu', W_regularizer=l2(weight_decay)))
model_py.add(Convolution2D(64, 64, 4, 4, activation='relu', W_regularizer=l2(weight_decay)))
model_py.add(MaxPooling2D(poolsize=(2, 3)))
model_py.add(Dropout(0.25))
model_py.add(Flatten())
model_py.add(Dense(64 * 7 * 10, 512, activation='relu', W_regularizer=l2(weight_decay)))
model_py.add(Dropout(0.5))
model_py.add(Dense(512, nb_class, activation='softmax'))

model_py.load_weights('./model/pinyin_model.d5')
model_py.compile(loss='categorical_crossentropy', optimizer='adagrad')

comp = "type1_test1"
img_dir = './image/' + comp + '/'
fcsv = codecs.open("./result/" + comp + '.csv', 'w', 'utf-8')
fcsv.write('pic_id,content\n')
# for nb_img in range(1,20001):  
#     name=comp+'_' + str(nb_img) + '.jpg' 
import os

names = os.listdir(img_dir)
for name in names:
    imgname = img_dir + name
    print imgname
    img = cv2.imread(imgname, cv2.IMREAD_GRAYSCALE)
    im = 255 - img
    retval, im = cv2.threshold(im, 255 * 0.3, 255, cv2.THRESH_BINARY)
    lines = cv2.HoughLinesP(im, 1, 0.5 * np.pi / 180, 50, maxLineGap=5, minLineLength=30)
    l = im - im
    for i in range(lines.shape[0]):
        for x1, y1, x2, y2 in lines[i]:
            cv2.line(l, (x1, y1), (x2, y2), 255, 1)
    im2 = im
    im = im - l
    x = [11, 78, 144] * 3
    y = [6, 6, 6, 46, 46, 46, 86, 86, 86]
    imgs = np.zeros((9, 1, 32, 32), np.float32)
    for i in range(9):
        imgs[i] = im2[y[i]:y[i] + 32, x[i]:x[i] + 32]
    scores1 = model_ch.predict(imgs / 255)
    classes1 = np.argsort(scores1, axis=1)
    result1 = []
    result1_2 = []
    result1_3 = []
    for c in range(classes1.shape[0]):
        result1.append(chars[classes1[c][-1]])
        result1_2.append(chars[classes1[c][-2]])
        result1_3.append(chars[classes1[c][-3]])
    print(''.join(result1))
    img = im2[125:157, 10:im.shape[1] - 15]
    t = im[125:157, 10:im.shape[1] - 15]
    t = cv2.medianBlur(t, 3)
    retval, t = cv2.threshold(t, 255 * 0.5, 1, cv2.THRESH_BINARY)
    s = np.sum(t[5:, :], axis=0)
    s2 = s.copy()
    s[s <= 4] = 0
    img = img[:, 0:max(110, min(s.nonzero()[0][-1] + 10, s.shape[0]))]
    idxl = list(range(0, img.shape[1] - 32, 30))
    idxr = map(lambda x: img.shape[1] - x - 32, idxl)
    idxl = idxl[:3]
    idxr = idxr[:3]
    idxs = idxl + idxr
    imgs = np.zeros((len(idxs), 1, 32, 32), np.float32)
    for i in range(len(idxs)):
        imgs[i] = img[0:32, idxs[i]:idxs[i] + 32]
    scores2 = model_ch.predict(imgs / 255)
    classes2 = np.argmax(scores2, axis=1)
    # ftrs2 = get_features(imgs / 255)
    result2 = []
    for c in classes2:
        result2.append(chars[c])
    print(''.join(result2))

    dd = scipy.spatial.distance.cdist(scores1, scores2, get_crossentropy)
    idx_up = dd.argmin(axis=0)
    ddm = dd.min(axis=0)
    for ii in range(len(result2)):
        if result2[ii] in result1:
            ddm[ii] = 0
            idx_up[ii] = result1.index(result2[ii])

    align_iterms = get_align_terms(idxl, idxr)
    for align_iterm in align_iterms:
        if ddm[align_iterm[0]] > ddm[align_iterm[1]]:
            ddm[align_iterm[0]] = np.inf
        else:
            ddm[align_iterm[1]] = np.inf

    idx_t = np.argsort(ddm)
    idx_t = idx_t[:3]
    idx_t = idx_t[np.asarray(idxs)[idx_t].argsort()]
    res = []
    for idx in idx_t:
        res.append(result2[idx])
    print(''.join(res))
    print(idx_up[idx_t] + 1)
    idx_list = list(idx_up[idx_t] + 1)
    py_pos = get_py_pos(np.asarray(idxs)[idx_t], img.shape[1])
    img_py = np.zeros((1, 1, 28, 90))
    img_py[0, 0, :, :py_pos[1] - py_pos[0]] = img[3:-1, py_pos[0]:py_pos[1]]
    classes_py = model_py.predict_classes(img_py / 255, verbose=0)
    pinyin = pys[classes_py]
    # print(pinyin)
    words = []
    words2 = []
    words3 = []
    words_idxs = []
    for i in range(9):
        if i not in idx_up[idx_t]:
            words.append(result1[i])
            words2.append(result1_2[i])
            words3.append(result1_3[i])
            words_idxs.append(i)
    wordspy = map(lambda w: ch_py[w], words)
    wordspy2 = map(lambda w: ch_py[w], words2)
    wordspy3 = map(lambda w: ch_py[w], words3)
    if pinyin in flat(wordspy):
        for ii in range(len(wordspy)):
            if pinyin in wordspy[ii]:
                pinyin_idx = ii
                break
    elif pinyin in flat(wordspy2):
        for ii in range(len(wordspy2)):
            if pinyin in wordspy2[ii]:
                pinyin_idx = ii
                break
    elif pinyin in flat(wordspy3):
        for ii in range(len(wordspy3)):
            if pinyin in wordspy3[ii]:
                pinyin_idx = ii
                break
    else:
        pinyin_idx = np.random.randint(0, len(wordspy))
    idx_list.insert(py_pos[0] / 30, words_idxs[pinyin_idx] + 1)
    # print(idx_list)
    fcsv.write(name + ',' + ''.join(map(lambda x: str(x), idx_list)) + '\n')
fcsv.close()

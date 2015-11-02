# coding=utf-8
# __author__ = 'lllcho'
# __date__ = '2015/10/1'

import cv2
import numpy as np
import codecs
import cPickle
import theano
from util import get_pos_media, words_simmilar_score, words_simmilar_score2
from scipy import signal
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.regularizers import l2

words_chars = cPickle.load(open('model/words_chars_type3.pkl', 'rb'))
chars1 = words_chars['chars1']
chars2 = words_chars['chars2']
words1 = words_chars['words1']
words2 = words_chars['words2']
word_szm = words_chars['word_szm']

weight_decay = 0.0005
chars1_model = Sequential()
chars1_model.add(Convolution2D(32, 1, 4, 4, border_mode='full', activation='relu', W_regularizer=l2(weight_decay)))
chars1_model.add(Convolution2D(32, 32, 4, 4, activation='relu', W_regularizer=l2(weight_decay)))
chars1_model.add(MaxPooling2D(poolsize=(2, 2)))
chars1_model.add(Dropout(0.25))
chars1_model.add(Convolution2D(64, 32, 4, 4, border_mode='full', activation='relu', W_regularizer=l2(weight_decay)))
chars1_model.add(Convolution2D(64, 64, 4, 4, activation='relu', W_regularizer=l2(weight_decay)))
chars1_model.add(MaxPooling2D(poolsize=(2, 2)))
chars1_model.add(Dropout(0.25))
chars1_model.add(Flatten())
chars1_model.add(Dense(64 * 8 * 8, 512, activation='relu', W_regularizer=l2(weight_decay)))
chars1_model.add(Dropout(0.5))
chars1_model.add(Dense(512, 1679, activation='softmax'))
chars1_model.load_weights('model/type3_chars1_model.d5')
chars1_model.compile(loss='categorical_crossentropy', optimizer='adagrad')
get_predict_score1 = theano.function([chars1_model.layers[0].input],
                                     chars1_model.layers[-1].get_output(train=False),
                                     allow_input_downcast=True)
chars2_model = Sequential()
chars2_model.add(Convolution2D(32, 1, 4, 4, border_mode='full', activation='relu', W_regularizer=l2(weight_decay)))
chars2_model.add(Convolution2D(32, 32, 4, 4, activation='relu', W_regularizer=l2(weight_decay)))
chars2_model.add(MaxPooling2D(poolsize=(2, 2)))
chars2_model.add(Dropout(0.25))
chars2_model.add(Convolution2D(64, 32, 4, 4, border_mode='full', activation='relu', W_regularizer=l2(weight_decay)))
chars2_model.add(Convolution2D(64, 64, 4, 4, activation='relu', W_regularizer=l2(weight_decay)))
chars2_model.add(MaxPooling2D(poolsize=(2, 2)))
chars2_model.add(Dropout(0.25))
chars2_model.add(Flatten())
chars2_model.add(Dense(64 * 8 * 8, 512, activation='relu', W_regularizer=l2(weight_decay)))
chars2_model.add(Dropout(0.5))
chars2_model.add(Dense(512, 1121, activation='softmax'))
chars2_model.load_weights('model/type3_chars2_model.d5')
chars2_model.compile(loss='categorical_crossentropy', optimizer='adagrad')
get_predict_score2 = theano.function([chars2_model.layers[0].input],
                                     chars2_model.layers[-1].get_output(train=False),
                                     allow_input_downcast=True)
nb_model = Sequential()
nb_model.add(Convolution2D(32, 1, 4, 4, border_mode='full', activation='relu', W_regularizer=l2(weight_decay)))
nb_model.add(Convolution2D(32, 32, 4, 4, activation='relu', W_regularizer=l2(weight_decay)))
nb_model.add(MaxPooling2D(poolsize=(2, 3)))
nb_model.add(Dropout(0.25))
nb_model.add(Convolution2D(64, 32, 4, 4, border_mode='full', activation='relu', W_regularizer=l2(weight_decay)))
nb_model.add(Convolution2D(64, 64, 4, 4, activation='relu', W_regularizer=l2(weight_decay)))
nb_model.add(MaxPooling2D(poolsize=(2, 3)))
nb_model.add(Dropout(0.25))
nb_model.add(Flatten())
nb_model.add(Dense(64 * 8 * 9, 512, activation='relu', W_regularizer=l2(weight_decay)))
nb_model.add(Dropout(0.5))
nb_model.add(Dense(512, 2, activation='softmax'))
nb_model.load_weights('model/type3_nbchars_model.d5')
nb_model.compile(loss='categorical_crossentropy', optimizer='adagrad')

comp = 'type3_test1'
img_dir = './image/' + comp + '/'
f_csv = codecs.open('./result/' + comp + '.csv', 'w', 'utf-8')
# for nb_img in range(1, 20001):
#     name = comp + '_' + str(nb_img) + '.png'
import os

names = os.listdir(img_dir)
for name in names:
    imgname = img_dir + name
    print imgname
    img = cv2.imread(imgname, cv2.IMREAD_COLOR)
    bg = cv2.imread('model/type3_bg.png', cv2.IMREAD_COLOR)
    img = img.astype(np.float32) - bg.astype(np.float32)
    img[img < 0] = 0
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    imt = np.asarray([0.3] * 60 + [0.2] * 65 + [0.08] * 125) * 255
    im = (img > imt).astype(np.uint8) * 255
    im0 = im.copy()
    im = im[10:-5, :170]
    if im.sum() / 255 > 750:
        # print (str(nb_img)+u"首字母")
        im = im[:, :85]
        nb_chars = nb_model.predict_classes(im.reshape((1, 1, 35, 85)).astype(np.float32) / 255, verbose=0)[0] + 4
        # print(nb_chars)
        imt = cv2.medianBlur(im, 3)
        s = imt.sum(axis=0)
        t = s.copy()
        t[t < 3 * 255] = 0
        start = min(15, max(t.nonzero()[0][3] - 3, 0))
        im = im[:, start:]

        media_pos = get_pos_media(im)
        media_pos[media_pos < 10] = 10
        win = signal.hann(30)
        media_pos_s = signal.convolve(media_pos, win, 'same') / sum(win)
        media_pos_s[media_pos_s < 10] = 10

        step = im.shape[1] / float(nb_chars)
        h_center = [int(i + step / 2) for i in np.arange(0, im.shape[1], step).tolist()]
        v_center = [int(media_pos_s[int(h)]) for h in h_center]
        h_center = [20 + start + h for h in h_center]
        v_center = [10 + v for v in v_center]
        # cv2.imshow('a',im)
        # cv2.waitKey()
        imt = np.zeros((im0.shape[0], im0.shape[1] + 20), im0.dtype)
        imt[:, 20:] = im0
        imgs = np.zeros((nb_chars, 1, 32, 32), np.uint8)
        kk = 0
        for v, h in zip(v_center, h_center):
            t = imt[v - 14:v + 14, h - 14:h + 14]
            imgs[kk, 0, :, :] = cv2.resize(t, (32, 32))
            kk += 1

        classes = chars2_model.predict_classes(imgs.astype(np.float32) / 255, verbose=0)
        model_predict_score = get_predict_score2(imgs.astype(np.float32) / 255)
        result = []
        for c in classes:
            result.append(chars2[c])
        word = ''.join(result)
        old_word = word
        if word not in words2:
            word_score = words_simmilar_score(word, words2)
            max_score = max(word_score.keys())
            if max_score > 0:
                candidate_words = word_score[max_score]
                if len(filter(lambda x: len(x) == len(old_word), candidate_words)) > 0:
                    candidate_words = filter(lambda x: len(x) == len(old_word), candidate_words)
                    predict_similar_score = {}
                    for candidate_word in candidate_words:
                        diff_chars = {}
                        for j in range(len(candidate_word)):
                            if old_word[j] != candidate_word[j]:
                                diff_chars[j] = candidate_word[j]
                        diff_chars_similar_score = 0
                        for key, item in diff_chars.items():
                            diff_chars_similar_score += model_predict_score[key, chars2.index(item)]
                        predict_similar_score[candidate_word] = diff_chars_similar_score
                    word = max(predict_similar_score.items(), key=lambda x: x[1])[0]
                else:
                    word = word_score[max_score][0]
            else:
                word = word_score[max_score][0]
        print word
        f_csv.write(name + ',' + word_szm[word] + '\n')
    else:
        # continue
        # print (str(nb_img)+u'成语')
        im = im[:, 75:170]
        imt = cv2.medianBlur(im, 3)
        s = imt.sum(axis=0)
        t = s.copy()
        t[t < 3 * 255] = 0
        start = min(10, max(t.nonzero()[0][3] - 5, 0))
        im = im[:, start:s.nonzero()[0][-1] + 3]
        media_pos = get_pos_media(im)
        media_pos[media_pos < 10] = 10
        win = signal.hann(30)
        media_pos_s = signal.convolve(media_pos, win, 'same') / sum(win)
        media_pos_s[media_pos_s < 10] = 10
        step = im.shape[1] / 4.0
        h_center = [int(i + step / 2) for i in np.arange(0, im.shape[1], step).tolist()]
        v_center = [int(media_pos_s[int(h)]) for h in h_center]
        h_center = [75 + start + h for h in h_center]
        v_center = [10 + v for v in v_center]
        imgs = np.zeros((4, 1, 32, 32), np.uint8)
        # cv2.imshow('a',im)
        # cv2.waitKey()
        kk = 0
        for v, h in zip(v_center, h_center):
            imt = im0[v - 14:v + 14, h - 14:h + 14]
            imgs[kk, 0, :, :] = cv2.resize(imt, (32, 32))
            kk += 1
        classes = chars1_model.predict_classes(imgs.astype(np.float32) / 255, verbose=0)
        model_predict_score = get_predict_score1(imgs.astype(np.float32) / 255)
        result = []
        for c in classes:
            result.append(chars1[c])
        word = ''.join(result)
        old_word = word
        if word not in words1:
            word_score = words_simmilar_score(word, words1)
            max_score = max(word_score.keys())
            if max_score > 1:
                candidate_words = word_score[max_score]
                if len(candidate_words) == 1:
                    word = candidate_words[0]
                else:
                    predict_similar_score = {}
                    for candidate_word in candidate_words:
                        diff_chars = {}
                        for j in range(min(len(candidate_word), len(old_word))):
                            if old_word[j] != candidate_word[j]:
                                diff_chars[j] = candidate_word[j]
                        diff_chars_similar_score = 0
                        for key, item in diff_chars.items():
                            diff_chars_similar_score += model_predict_score[key, chars1.index(item)]
                        predict_similar_score[candidate_word] = diff_chars_similar_score
                    word = max(predict_similar_score.items(), key=lambda x: x[1])[0]
        print word
        f_csv.write(name + ',' + word + '\n')
f_csv.close()

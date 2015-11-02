# coding:utf-8
# __author__ = 'lllcho'
# __date__ = '2015/8/4'
import cv2
import cPickle
import numpy as np
import codecs
import h5py
import theano
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D


def word_simialr_score(s1, s2):
    score = 0
    for j in range(len(s1)):
        if s1[j] == s2[j]:
            score += 1
    return score


def words_simmilar_score(word, words):
    word_score = {}
    for Word in words:
        ws = word_simialr_score(word, Word)
        if ws not in word_score.keys():
            word_score[ws] = [Word]
        else:
            word_score[ws].append(Word)
    return word_score


np.random.seed(123)
model_path = './model/type4_model.d5'

chars = cPickle.load(open('model/chars_type4.pkl', 'rb'))
words = cPickle.load(open('model/words_type4.pkl', 'rb'))
chars.append('A')

f = h5py.File('./model/type4_train_mean_std.h5', 'r')
x_mean = f['x_mean'][:]
x_std = f['x_std'][:][0]
f.close()
model = Sequential()
model.add(Convolution2D(32, 3, 4, 4, border_mode='full', activation='relu'))
model.add(Convolution2D(32, 32, 4, 4, activation='relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))
model.add(Convolution2D(64, 32, 4, 4, border_mode='full', activation='relu'))
model.add(Convolution2D(64, 64, 4, 4, activation='relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64 * 8 * 8, 512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, 1250, activation='softmax'))
model.load_weights(model_path)
model.compile(loss='categorical_crossentropy', optimizer='adagrad')
get_predict_score = theano.function([model.layers[0].input],
                                    model.layers[-1].get_output(train=False),
                                    allow_input_downcast=True)
comp = 'type4_test1'
img_dir = './image/' + comp + '/'
fcsv = codecs.open("result/" + comp + '.csv', 'w', 'utf-8')
# for nb_img in range(1, 20001):
#     name=comp+'_'+str(nb_img)+'.png'
import os

names = os.listdir(img_dir)
for name in names:
    print name
    imgname = img_dir + name
    img = cv2.imread(imgname, cv2.IMREAD_COLOR)
    im = 255 - img[4:-4, :, :]
    t = im.shape[1] / 4.0
    dd = 5
    bb = np.zeros((im.shape[0], dd, 3), dtype=np.uint8)
    im1 = im[:, 0:np.floor(t) + dd]
    im2 = im[:, np.floor(t) - dd:np.floor(2 * t) + dd]
    im3 = im[:, np.floor(2 * t) - dd:np.floor(3 * t) + dd]
    im4 = im[:, np.floor(3 * t) - dd:]
    imgs = np.zeros((4, 3, 32, 32))
    imgs[0, :] = cv2.resize(np.concatenate((bb, im1), axis=1), (32, 32)).transpose()
    imgs[1, :] = cv2.resize(im2, (32, 32)).transpose()
    imgs[2, :] = cv2.resize(im3, (32, 32)).transpose()
    imgs[3, :] = cv2.resize(np.concatenate((im4, bb), axis=1), (32, 32)).transpose()

    imgs.astype(np.float32)
    imgs = imgs - x_mean
    imgs = imgs / x_std
    classes = model.predict_classes(imgs, verbose=0)
    model_predict_score = get_predict_score(imgs)
    result = []
    for c in classes:
        result.append(chars[c])
    word = ''.join(result)
    old_word = word
    if word not in words:
        word_score = words_simmilar_score(word, words)
        max_score = max(word_score.keys())
        if max_score > 0:
            candidate_words = word_score[max_score]
            predict_similar_score = {}
            for candidate_word in candidate_words:
                diff_chars = {}
                for j in range(len(candidate_word)):
                    if old_word[j] != candidate_word[j]:
                        diff_chars[j] = candidate_word[j]
                diff_chars_similar_score = 0
                for key, iterm in diff_chars.items():
                    diff_chars_similar_score += model_predict_score[key, chars.index(iterm)]
                predict_similar_score[candidate_word] = diff_chars_similar_score
            word = max(predict_similar_score.items(), key=lambda x: x[1])[0]
    print word
    fcsv.write(name + ',' + word + '\n')

fcsv.close()

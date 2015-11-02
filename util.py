# __author__ = 'lllcho'
# __date__ = '2015/8/2'
import math
import numpy as np
import theano
import theano.tensor as T

flat = lambda L: sum(map(flat, L), []) if isinstance(L, list) else [L]
s1 = T.vector('s1')
s2 = T.vector('s2')
ce = T.nnet.categorical_crossentropy(s1, s2)
ccee = theano.function([s1, s2], ce, allow_input_downcast=True)


def get_align_terms(l, r, margin=32):
    res = []
    for i in range(len(l)):
        for j in range(len(r)):
            if abs(l[i] - r[j]) < margin * 0.3:
                res.append((i, j + len(l)))
    return res


def get_py_pos(idxs, end):
    assert len(idxs) == 3
    if idxs[0] >= 5:
        return 0, idxs[0]
    elif idxs[1] - 30 >= 5:
        return 30, idxs[1]
    elif idxs[2] - 60 >= 5:
        return 60, idxs[2]
    else:
        return 90, end


def get_crossentropy(x, y):
    return ccee(x, y)


def get_pos_media(img):
    img = img > 1
    s = np.zeros((img.shape[1]), dtype=np.uint8)
    for i in range(s.shape[0]):
        nz = np.median(img[:, i].nonzero()[0])
        if math.isnan(nz):
            nz = img.shape[0] / 2
        s[i] = nz
    return s


def word_simialr_score(s1, s2):
    score = 0
    for j in range(min(len(s1), len(s2))):
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


def word_simialr_score2(s1, s2):
    score = len(set(list(s1)) & set(list(s2)))
    return score


def words_simmilar_score2(word, words):
    word_score = {}
    for Word in words:
        ws = word_simialr_score2(word, Word)
        if ws not in word_score.keys():
            word_score[ws] = [Word]
        else:
            word_score[ws].append(Word)
    return word_score

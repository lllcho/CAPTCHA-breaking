# coding:utf-8
# __author__ = 'lllcho'
# __date__ = '2015/7/30'
import glob
import os
import numpy as np
from PIL import Image
imgnames=glob.glob('E:/DC/data/train/image/letter_img/*.jpg')
for imgname in imgnames:
    img=np.asarray(Image.open(imgname))

pass
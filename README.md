# Testing codes for [CAPTCHA Breaking Challenge (DataCastle)](http://www.pkbigdata.com/common/competition/104.html)  

## [Thoughts behind (Chinese)](http://bbs.pkbigdata.com/topic/e0286749ba3245bc96444122cc9877db.html)  

## Installation  
**Testing environment**  
ubuntu14.04 + python2.7  

**Direct dependencies**  
- keras (required version of theano : 0.7.1)  
- python-opencv  

**Installation pipeline recommended**  
1 Install [Anaconda](https://www.continuum.io/downloads#_unix) ([alternative](http://pan.baidu.com/s/1eQsHswM)) to replace default python;  

2 Configure CUDA and alter ```~/.theanorc``` (ignore the step if not use GPU);
```
vim ~/.theanorc
```

Then add:  
```
[global]
  device = gpu
  floatX = float32
```
  
3 Install keras: using the default or specified version ([0.1.2](https://github.com/fchollet/keras/archive/0.1.2.tar.gz)) to avoid the problems caused by changed API of the latest version;
```
cd keras-master  
python setup.py install
```
  
4 Configure openCV: ```sudo apt-get install python-opencv```, then add cv2.so to the python path.  

*Note: the codes work for windows as well, just make sure the python dependencies have been installed.*  

## How to use  
1 Change the input (where the CAPTCHA pictures are stored) and output path in the script;  
2 run ```python test_type*N*.py```.  

## Testing results  
| Type | Accuracy |
|------|----|
|type1 |0.92|
|type2 |0.99|
|type3 |0.99|
|type4 |1   |
|type5 |0.74|
|type6 |0.37|  

*Find more testing data [here](http://pan.baidu.com/s/1hqk6rxa)*  
  
---  

# 中文说明

## 安装
测试环境：ubuntu14.04+python2.7

直接依赖：

+ keras(theano版本要求0.7.1)
+ python-opencv

推荐安装步骤：

1. 安装 [Anaconda](https://www.continuum.io/downloads#_unix) 或者点击[这里](http://pan.baidu.com/s/1eQsHswM)下载安装并取代系统默认Python
2. 配置CUDA并修改`~/.theanorc`文件(不使用gpu可省略这一步)
> `vim ~/.theanorc`
> 然后添加：
>  `[global]`
    `device=gpu`
    `floatX=float32`

3. 由于最新版keras API发生改变，请使用自带版本，或下载指定版本([0.1.2](https://github.com/fchollet/keras/archive/0.1.2.tar.gz))
> `cd keras-master`
> `python setup.py install`
4. 配置opencv: `sudo apt-get install python-opencv` 然后将cv2.so添加到PYTHON路径

代码在windows下也测试通过，安装windows版本python依赖即可

## 使用
1. 打开脚本修改验证码图片所在路径以及结果输出路径
2. 运行命令`python test_type*N*.py`

## 测试结果
| 类型 | 识别率 |
|------|----|
|type1 |0.92|
|type2 |0.99|
|type3 |0.99|
|type4 |1   |
|type5 |0.74|
|type6 |0.37|
更多测试数据[下载地址](http://pan.baidu.com/s/1hqk6rxa)

# [DataCastle验证码识别大赛](http://www.pkbigdata.com/common/competition/104.html)测试源码

## 安装
测试环境：ubuntu14.04+python2.7
直接依赖：

+ keras
+ python-opencv

推荐安装步骤：

+ 安装 [Anaconda](https://www.continuum.io/downloads#_unix) 或者点击[这里](http://pan.baidu.com/s/1eQsHswM)下载安装并取代系统默认Python
+ 配置CUDA并修改`~/.theanorc`文件(不使用gpu可省略这一步)
> `vim ~/.theanorc`
> 然后添加：
>  `[global]`
    `device=gpu`
    `floatX=float32`

+ 由于最新版keras API发生改变，请使用自带版本，或下载指定版本([0.1.2](https://github.com/fchollet/keras/archive/0.1.2.tar.gz))
> `cd keras-master`
> `python setup.py install`
+ 配置opencv: `sudo apt-get install python-opencv` 然后将cv2.so添加到PYTHON路径

代码在windows下也测试通过，安装windows版本python依赖即可

## 使用
+ 打开脚本修改验证码图片所在路径以及结果输出路径
+ 运行命令`python test_type*N*.py`

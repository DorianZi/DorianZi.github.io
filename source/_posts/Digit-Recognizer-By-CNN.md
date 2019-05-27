---
title: Digit Recognizer by CNN 基于卷积神经网络的手写数字识别
date: 2019-05-01 08:17:19
tags: ["Algorithm", "Deep Learning", "TensorFlow"]
categories: Technic
---

本文将用TensorFlow来实现CNN经典的入门项目:手写数字识别

# 数据集

![](/uploads/digit_recognizer_1.png)

手写数字的数据集来自著名的MNIST(美国国家标准与技术研究所)，它包含了6万个训练样本和1万个测试样本，并且所有样本都已经标准化为28\*28个像素，每个像素值在0~1之间。同时每张图片的储存方式已经扁平化为784（28\*28）个元素的一维numpy序列。

# 网络结构

![](/uploads/digit_recognizer_2.png)

我们要构建的CNN网络大致如上，它实际上是使用了[LeNet](https://dorianzi.github.io/2019/04/25/CNN/#LeNet-1986):

![](/uploads/digit_recognizer_3.png)

# 代码解析

完整代码[见此](https://github.com/DorianZi/kaggle/blob/master/digit_recognizer/digit_recognizer.py)

## 安装TensorFlow

```
$ sudo pip install tensorflow
如果用到CUDA则为：
$ sudo pip install tensorflow-gpu

以上命令会按照时候本机的最新版本。如果要安装指定版本（可能不兼容），使用：
$ sudo pip install tensorflow-gpu==1.70

查看安装的版本：
$ pip freeze
```

## 获得数据集

开始我们的coding.

### 从TensorFlow获取
首先，获取数据集，可以通过tensorflow官方api直接拿到mnist数据集

```
# 导入mnist数据集
from tensorflow.examples.tutorials.mnist import input_data

# 数据压缩包下载保存到/tmp/data下
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

# 分别在得到train和test数据集，images的类型为numpy.ndarray
print(mnist.train.images)
print(mnist.test.images)
```

###  从Kaggle比赛项目获取
本文中的完整代码是Kaggle比赛的项目，数据集在该[比赛项目页面](https://www.kaggle.com/c/digit-recognizer/data)可以本地下载获得。下载得到的训练集为train.csv，测试集为test.csv

train.csv的格式如下，test.csv相比train.csv少了label列

```
label    pixel0    pixel1    pixel2    pixel3    ...   pixel783
1          0         255       0         0       ...      0
4          0         0         0         0       ...      0
0          0         0         98        0       ...      0
...
7          0         0         0         0       ...      0

```
以上格式的csv可以通过pandas模块的read_csv读取：

```
import pandas as pd

# 读取csv文件为DataFrame
csvDF = pd.read_csv("train.csv")

# 将DataFrame中的'label'列去掉
train_data = csvDF.drop(columns = ['label'])

# 去掉表头，拿到纯数据，也就是格式为numpy.ndarray的数据
train_data = train_data.values

# 将原shape为(42000,784)的数据变换为(42000,28,28,1)的数据，并且将每个数据由原来的numpy.int64类型变成numpy.float32类型
train_data = train_data.reshape(-1,28,28,1).astype('float32')

# 获取'label'列数据，此时为Series类型
train_labels = csvDF['label']

# 获取纯数据，numpy.ndarray类型
train_labels = train_labels.values

# 引入sklearn模块
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# 进行标签编码， 这里和train_labels = train_labels.reshape(-1,1)是一样的
train_labels = LabelEncoder().fit_transform(train_labels)[:, None]

# 进行独热编码
train_labels = OneHotEncoder().fit_transform(train_labels).todense()

# 读取test的csv文件
csvDF = pd.read_csv("test.csv")

# 数据储存到shape是(28000,28,28,1)，每个数据类型为numpy.float32的numpy.ndarray变量中
test_data = csvDF.values.reshape(-1,28,28,1).astype('float32')
```

### 代码解析

#### pandas中的数据结构

上面提到的DataFrame是pandas定义的一种格式，是一个类，这里csvDF是它的一个实例。在pandas常用的数据格式是三种：它自己的DataFrame,Series和numpy的ndarray。他们之间的关系可以描述如下：

DataFrame可以通过字典方式访问每一列，如csvDF['label']或csvDF.label代表'label'所在的列，csvDF['label']或csvDF.label就是Series格式的

对于DataFrame来说，直接取数值（即去掉表头）: csvDF.values就得到了二维的numpy.ndarray格式数据；同理对于Series来说也可以去掉标注信息：csvDF['label'].values 获得一维度的numpy.ndarray格式数据。

#### 多维数据在numpy.ndarray中的表示

numpy.ndarray中的多维数据是通过列表嵌套得到的。

表示一个shape为(9,)的数据：[1,2,3,4,5,6,7,8,9]
reshape为(3,3)的数据：[[1,2,3], [4,5,6], [7,8,9]]
reshape为(3,3,1)的数据：[[[1],[2],[3]], [[4],[5],[6]], [[7],[8],[9]]]

#### 独热编码

（蓝色，红色，黄色）可以标签编码为(1,2,3)，但是这样在计算上会造成 “(蓝色+黄色)/2=红色” 的不良后果。通过独热编码将它们映射到欧式空间可以解决这个问题：

蓝色=(1,0,0)
红色=(0,1,0)
黄色=(0,0,1)

在本文中，我们的label已经是0~9的数字，可以作为标签编码使用。但是考虑到神经网络最后有10个输出，为独热编码形式，所以这里我们使用独热编码来表示。


### 验证集分割

将训练集分割为实际训练集和验证集，为常用的训练前数据的准备方法。我们的训练过程通过考察该验证集的预测准确率来决定是否停止，而不是考察实际训练集的准确率。这样可以有效地减少过拟合可能。

```
import numpy as np

# 拿出1000个数据作为验证集
VALID = 1000

# 抽取出前1000个数据出来作为验证集
valid_data = train_data[0:VALID]

# 删掉验证集，剩下的就是实际训练集
train_data = np.delete(train_data,np.s_[0:VALID],0)
```

上面代码中np.delete接受的三个参数分别为：numpy.ndarray类型的待切割数据集，list类型的被删除的行（列）的列表，指定删除行（0）或列（1）

而上面使用的np.s\_[0:1000]有特殊的功能，它是指取得指定范围的标号列表，也就是[0,1,2,3,...,999]

## 使用TensorFlow构建CNN网络

TensorFlow构建网络的方式大体上分两个步骤：1）先通过创建占位符号，构建图 2）然后在运行图的时候将数据喂进去，实现训练和预测。
构建图的过程是设计网络的过程，这中间用到的数据变量，都是“空壳”，它们占用了内存，但是里面并没有数据。运行的图的时候一定要喂数据，否则空图是不能运行处结果的。

```
import tensorflow as tf

# 定义输入数据的占位符，shape为（None,28,28,1）,其中None表示可以是任何数，数据类型为tf.float32，其实和np.float32没区别
# 被定义的x变量在之后会被喂进上面的训练数据，而None之所以不设为1，是为了批量训练（一次性喂多行数据）准备的
x = tf.placeholder(tf.float32, [None,28,28,1])

# 输入label的占位符，且该label已经是独热编码形式，在训练的时候会被喂入train_labels的每一行数据
y = tf.placeholder(tf.float32,[None,10])

# 定义5*5的卷积核，其数值为随机数，shape为(5,5,1,32)
w1 = tf.Variable(tf.random_normal([5,5,1,32]))

# 定义偏置，随机初始化
b1 = tf.Variable(tf.random_normal([32]))

# 第二次卷积的卷积核
w2 = tf.Variable(tf.random_normal([5,5,32,64]))

# 第二次卷积的偏置
b2 = tf.Variable(tf.random_normal([64]))

# 第三次卷积的卷积核
w3 = tf.Variable(tf.random_normal([5,5,64,64]))

# 第三次卷积的偏置
b3 = tf.Variable(tf.random_normal([64]))

# 第一次全连接的权重
wf_1 = tf.Variable(tf.random_normal([4*4*64,1024]))

# 第一次全连接的偏置
bf_1 = tf.Variable(tf.random_normal([1024]))

# dropout里的保留率
keep_prob = tf.placeholder(tf.float32)

# 第二次全连接的权重
wf_2 = tf.Variable(tf.random_normal([1024,10]))

# 第二次全连接的偏置
bf_2 = tf.Variable(tf.random_normal([10]))

# 梯度下降算法的学习率
learn_rate = tf.placeholder(tf.float32)


def conv2d(X,W,b,k=2):
    # 二维卷积
    conv = tf.nn.conv2d(X,W,strides=[1,1,1,1], padding='SAME')
    # 加偏置。将b中第i个元素加到conv中第i个通道里所有元素上
    conv = tf.nn.bias_add(conv,b)
    # 通过relu激活每一个元素（即保留正数），并且不改变conv的shape
    conv = tf.nn.relu(conv)
    # 最大池化
    conv = tf.nn.max_pool(conv, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
    return conv

# 连续三次conv -> bias -> relue -> pooling 操作
co1 = conv2d(x, w1, b1, k=2)                 # co1的shape为(?,14,14,32)
co2 = conv2d(co1, w2, b2, k=2)               # co2的shape为(?,7,7,64)
co3 = conv2d(co2, w3, b3, k=2)               # co3的shape为(?,4,4,64)，因为池化时有补零，所以7->4

# 扁平化为（?,4*4*64)，为全连接作准备
co3 = tf.reshape(co3,[-1,4*4*64])            # 这里用-1而不用None，是因为这里不是初始化，这里的cos3已经被分配空间了

# 第一次全连接并用relu激活。tf.matmul为矩阵乘法，这里计算的是shape变化为（?,4*4*64) *（4*4*64,1024) = (?,1024)
fc = tf.nn.relu(tf.matmul(co3,wf_1) + bf_1)  # fc的shape为(?,1024)

# 以keep_prob的保留概率进行dropout，防止过拟合
fc_keep = tf.nn.dropout(fc, keep_prob)

# 第二次全连接，以获得每个数据的10位编码，为softmax进行归一概率化做准备
logits  = tf.matmul(fc_keep, wf_2) + bf_2    # logits的shape为(?,10) = (?,1024) * (1024,10)

# 使用softmax进行独热分类预测，输出值shape为(?,10)，它的每一个元素代表预测出来的对应类别的概率
prediction = tf.nn.softmax(logits)

# 计算对shape为(?,10)计算每一行的交叉熵得到shape为（?,1）的数据，然后计算每个元素的均值得到单一数值loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))

# 使用随机梯度下降的优化算法Adam对loss进行优化
optimizer = tf.train.AdamOptimizer(learn_rate).minimize(loss)

# tf.argmax(a,axis=1)按行获得最大值的下标号,输出shape为(?,1)，这么做是因为最大值的那个类别才是被认为预测的分类
# tf.equal(A,B)按元素获取bool值，输出shape为A.shape以及B.shape。此例中为(?,1)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

# 用tf.cast将每个位置的数值转换为tf.float32类型，然后再求均值。
# 这里的均值之所以可以表示准确度是因为correct_pred的每一位是非0即1的值，求准确度就是求1出现的频率，也恰好是求均值
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

```

### 代码讲解

#### tf.nn.conv2d函数
conv2函数里的参数strides表示卷积核的扫描步长，两边两个一般默认为1(因为不用对batch和channel方向做大于1的步长扫描)，中间两个代表纵向和横向扫描的步长。如strides=[1,4,5,1]表示卷积核纵向一次移动4个元素，横向一次移动5个元素。

conv2函数里的参数padding='SAME'代表通过补零方式，保证卷积输出的数据和原数据shape保持一致，否则padding='VALID'表示不补零，数据输出将有shape损失。它的效果在之前的文章中有过[介绍](https://dorianzi.github.io/2019/04/25/CNN/#Padding)

当被卷积数据和卷积核的shape分别为（None,28,28,1），(5,5,1,32)，它们是怎么被conv2d函数完成卷积的呢？

在conv2d函数看来，（None,28,28,1）中第一个维度None被认为是batch数，喂数据的时候喂入多行数据的时候就是多个卷积操作平行进行，互不干涉。第二、三个维度被认为是二维数据的高和宽，也就是数据为28\*28形状。第四个维度被认为是数据的通道数，比如一张shape为(2,2,3)的彩色图片数据为[[[123,111,88], [123,234,87]], [[88,65,29], [76,20,246]]]，它是2\*2大小，而每个元素又分成三部分，每部分是RGB的一个值。

(5,5,1,32)前两个维度被认为是每个卷积核的形状，第三个维度的值1是输入通道数，对应的是被卷积数据的通道数，而第四个维度的值32是卷积后输出的通道数。

综合描述（None,28,28,1）和(5,5,1,32)的conv2d操作，就是对多行28\*28的1通道的图片，使用1\*32个5\*5的卷积核进行卷积。

图解卷积发生的方式：

![](/uploads/digit_recognizer_4.png)

这里要着重理解一下卷积的分配方式和求和方式。思考一个问题：当我们把4个通道的图片进行conv2d操作之后变成了3个通道，这个过程中怎么对这4个通道都“公平”，并且求出来的3个通道也是公平的？

CNN的设计中实现了公平：对每个通道用不同的卷积核进行卷积，然后求和，这样形成一个新的通道。以上过程进行3次，得到3个通道。

之所以说对4个通道是公平的，是因为所有4个通道都被卷积，且没有一个通道被使用了特殊的卷积核，大家都是用随机的不同的卷积核，卷积完了进行求和，没有哪个的权重更大。

之所以说对生成的3个通道是公平的，是因为以上过程进行的3次重复，没有哪次的权重是更多的。

如果你能创造新的“公平”方式且能使得CNN的预测结果得到提升，那么恭喜你，你创造了一种CNN的改进算法。

#### tf.nn.bias_add函数

它是tf.nn.add的一种特例。tf.nn.add(x,y)的第二个参数y可以是单数值，这样话，该数值可以广播并相加到矩阵的任何一个元素上。
tf.nn.bias_add(x,y)的第二个参数y的维度必须跟x的最后一个维度是一致的

#### tf.nn.max_pool函数

进行最大池化的函数tf.nn.max_pool(value, ksize, strides, padding, name=None)

value为被池化对象；ksize=[1,height,width,1]表示池化窗口，因为不想在batch和channels上做池化，所以这两个维度设为了1；strides跟卷积一样，是每个维度上的步长；padding跟卷积一样，表示是否补零

比如shape为（1,4,4,2）的图片数据A，为：
![](/uploads/digit_recognizer_5.png)
![](/uploads/digit_recognizer_6.png)

经过池化操作tf.nn.max_pool(A,[1,2,2,1],[1,1,1,1],padding='VALID')之后得到shape为（1,3,3,2）的图片数据：
![](/uploads/digit_recognizer_7.png)
![](/uploads/digit_recognizer_8.png)

#### tf.nn.softmax函数

softmax对矩阵的每一行进行如下归一化操作，使得每一位都是相对于该行的概率(大小在0~1之间)：
![](/uploads/digit_recognizer_9.png)

#### tf.nn.softmax_cross_entropy_with_logits_v2函数

tf.nn.softmax_cross_entropy_with_logits(logits, labels, name=None)用来计算交叉熵。

第一个参数logits是预测的分布，第二个参数label是真实的分布，他们的shape为（batchsize，num_classes），输出的shape为（batchsize，1）

交叉熵的公式是：

<img src="https://latex.codecogs.com/gif.latex?H(p,q)=-\sum_{i=1}^np(x_i)log(q(x_i))" />

其中<img src="https://latex.codecogs.com/gif.latex?p(x_i)" />为真实的概率分布，<img src="https://latex.codecogs.com/gif.latex?q(x_i)" />为预测的概率分布。

预测的概率分布越接近真实概率分布，则交叉熵越小。

## 训练和预测

```
# 创建TensorFlow的会话，相当于初始化。
sess = tf.Session()

# 构建关于所有全局变量初始化的计算图，这样就不需要一个一个地去初始化之前定义的变量
init = tf.global_variables_initializer()

# 运行init计算图，将所有定义的变量初始化
sess.run(init)

# 将训练数据和label进行乱序，然后生成队列，每次可以从队列出队1000个数据
# 这里定义的还是队列图，需要用tf.train.start_queue_runners使得队列图运转起来
data_queue, labels_queue = tf.train.shuffle_batch([train_data,train_labels],
                                                   batch_size=1000,
                                                   capacity=50000,
                                                   min_after_dequeue=10000,
                                                   enqueue_many=True)


# 创建协调器用来协调线程
coord = tf.train.Coordinator()

# start_queue_runners会启动多线程，需要用coord对线程进行协调，当然最基础的还是需要指定sess，tf所有操作都必须在session中
# 启动多线程的原因是tf运行过程中，数据（并非针对上面的shuffle_batch）的读取和计算需要同时进行，而不是计算的时候，取数据就停滞
threads = tf.train.start_queue_runners(sess,coord)

# 每个epoch的迭代次数
steps_per_epoch = int(train_data.shape[0]/train_batch_size)

# 训练200轮，每一轮意味着过完一遍训练数据
epochs = 200

# 总迭代数
steps = steps_per_epoch * epochs

print ("All steps:  {0}*{1} = {2}".format(steps_per_epoch,epochs,steps))

# early stop标志位
earlyStopProc = False

# 辅助early stop标志位
noImproveCount = 0

# 当前epoch
cur_epoch = 1

# 初始梯度下降的学习率
initial_learn_rate = 0.001

# 当前学习率
cur_learn_rate = 0.001

for step in range(1,steps+1):
    # 获取本轮的batch数据
    batch_x, batch_y = sess.run([data_queue, labels_queue])
    
    # 将数据喂入优化器进行梯度下降优化，结果是更新了optimizer图中的权重，偏置
    sess.run(optimizer,feed_dict={x:batch_x,
                                  y:batch_y,
                                  learn_rate: cur_learn_rate,
                                  keep_prob: 0.75})
    
    # 每100次进行状态检查
    if step % 100 == 0 or step == 1:

        # 使用上次sess.run已更新的权重和偏置，再次喂入本轮batch以计算损失和训练数据准确率。计算过程使用100%的数据，不dropout
        lossval, accval = sess.run([loss, accuracy], feed_dict={x: batch_x,
                                                                y: batch_y,
                                                                keep_prob: 1.0})   # 计算过程使用100%的数据，不dropout

        # 喂入验证数据，计算验证集准确率，计算过程使用100%的数据，不dropout
        validacc = sess.run(accuracy, feed_dict={x: valid_data,
                                                 y: valid_labels,
                                                 keep_prob: 1.0})

        print ("Epoch {0}/{1}, Step {2}/{3}: Learn Rate={4}, Minibatch Loss={5}, Training Accuracy={6}, Validation Accuracy={7}".format(cur_epoch,epochs,step,steps,cur_learn_rate,lossval,accval,validacc))

        # //表示整数除法，保留整数位
        cur_epoch = step//steps_per_epoch + 1
        if cur_epoch > 1:
            # 每过10个epoch，就将学习率衰减到当前学习率的0.9
            cur_learn_rate = initial_learn_rate * pow(0.9,(cur_epoch-1)//10)

        # 在训练大于100个epoch的前提下当验证集准确率大于0.993则立即停止训练
        # 若没大于0.993但近500次的验证集准确率都没有提升，则也停止训练
        if earlyStopProc:
            if validacc > 0.993:
                print("validacc > 0.993 in latest 500. Early Stopping !")
                break
            noImproveCount += 1
            if noImproveCount > 5:
                noImproveCount = 0
                earlyStopProc = False
        if validacc > 0.993 and cur_epoch > 100:
            earlyStopProc = True

        # 训练大于20个epoch，验证集的准确率还是低于0.9，则本次训练不正常（可能参数初始化导致梯度消失），终止训练
        if cur_epoch > 20 and validacc < 0.9:
            print("Broken. Stop!")
            break
        print ("Trainning Finished!")

        # 对所有线程发送终止请求
        coord.request_stop()
        coord.join(threads)

# 对测试机数据进行分类预测。通过tf.argmax将独热编码prediction（它的shape为(?,10)）每一行的下标号取出，即为分类数字
# 预测的时候喂入测试数据集，并且不使用dropout，返回的pred_number的shape为(?,1)
pred_number = sess.run(tf.argmax(prediction,1),
                       feed_dict={x:test_data, keep_prob: 1.0})
# 关闭会话
sess.close()
```

### 代码讲解

#### sess.run函数

sess.run(arg0, feed_dict=dict)第一个参数arg0为构建好的图或者多个图的列表，我们要run的就是这个arg0，run出结果之后函数返回也是这个arg0的计算值，feed_dict指定在运行过程喂数据的路径

#### tf.train.shuffle_batch函数

tf.train.shuffle_batch([train_data,train_labels],batch_size=1000, capacity=50000,min_after_dequeue=10000, enqueue_many=True) 工作方式为：

从420000条总数据里面，取出50000条数据组成一个小队列，并且打乱顺序，从队尾取出1000条数据，然后从总数据里面拿1000条过来补充到队头，再次打乱顺序，然后继续从队尾取出1000条，然后再打乱，再补充......当补充进来的数据超过了总数据，就再回到总数据的开始继续拿数据补充。

min_after_dequeue起的作用是要求capacity被取走batch_size的数据之后，队列里剩下的数据量要不小于min_after_dequeue


## 数据增强

在**训练之前**，我们可以通过Data augmentation获得更多的数据。数据更多当然就更有利于训练出精度更高，泛化能力更强的模型。

见代码：
```
# 使用keras里的ImageDataGenerator函数进行图片增广
from keras.preprocessing.image import ImageDataGenerator

# 导入绘图工具
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def augTrainData(train_data, aug_times=1):
    print ("Before augmentaion: {0}, {1}".format(train_data.shape,train_labels.shape))
    datagen = ImageDataGenerator(rotation_range=10,         # 随机旋转10度
                                 zoom_range = 0.1,          # 随机缩放范围是±0.1
                                 width_shift_range=0.1,     # 随机水平移动范围是±0.1
                                 height_shift_range=0.1)    # 随机上下移动范围是±0.1

    # 载入数据，生成迭代器，一次输出len(train_data)张图片,通过next()进行迭代
    imgs_flow = datagen.flow(train_data.copy(), batch_size=len(train_data), shuffle = False)

    # label直接复制，无需进行变换
    labels_aug = train_labels.copy()

    # 将数据扩展aug_times倍
    for i in range(aug_times):
        # 取出len(train_data)张图片
        imgs_aug = imgs_flow.next()
        # 增加行
        train_data = np.append(train_data,imgs_aug,0)
        train_labels = np.append(train_labels,labels_aug,0)
    print ("After augmentaion: {0}, {1}".format(train_data.shape,train_labels.shape))
    try:
        print ("Trying to show augmentation effects")

        # 用subplots绘制多个子图
        # fig为整个图的对象，axs为多个（5*10）子图对象的二维列表，每个子图的高为15pixel，宽为9pixel
        fig,axs = plt.subplots(5,10, figsize=(15,9))
        
        # 对不同位置子图绘制图片
        axs[0,1].imshow(train_data[len(train_data)//1000*1].reshape(28,28), cmap=cm.binary)
        axs[0,2].imshow(train_data[len(train_data)//1000*500].reshape(28,28), cmap=cm.binary)
        axs[0,3].imshow(train_data[len(train_data)//1000*999].reshape(28,28), cmap=cm.binary)
        
        # 展示图片
        plt.show()
    except:
        print ("No X server, skipping drawing")

train_data = augTrainData(train_data,1)
```


## 训练效果
```
Before augmentaion: (32000, 28, 28, 1), (32000, 10)
After augmentaion: (192000, 28, 28, 1), (192000, 10)
...
All steps:  1920*200 = 384000
Epoch 1/200, Step 1/384000: Learn Rate=0.001, Minibatch Loss=1778990.125, Training Accuracy=0.109999999404, Validation Accuracy=0.132499992847
Epoch 1/200, Step 100/384000: Learn Rate=0.001, Minibatch Loss=46582.765625, Training Accuracy=0.680000007153, Validation Accuracy=0.77120000124
Epoch 1/200, Step 200/384000: Learn Rate=0.001, Minibatch Loss=30213.3691406, Training Accuracy=0.759999990463, Validation Accuracy=0.851599991322
...
Epoch 120/200, Step 230300/384000: Learn Rate=0.00031381059609, Minibatch Loss=0.0, Training Accuracy=1.0, Validation Accuracy=0.992200016975
43813 Epoch 120/200, Step 230400/384000: Learn Rate=0.00031381059609, Minibatch Loss=0.0, Training Accuracy=1.0, Validation Accuracy=0.992699980736
43814 Epoch 121/200, Step 230500/384000: Learn Rate=0.000282429536481, Minibatch Loss=0.0, Training Accuracy=1.0, Validation Accuracy=0.992200016975
...


```

以上
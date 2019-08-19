---
title: Official YOLOv3 training on VOC 使用官方YOLOv3训练VOC数据集
date: 2019-08-05 07:07:15
tags: ["Deep Learning", "Computer Vision"]
categories: Technic
---

在[之前的文章](https://dorianzi.github.io/2019/07/05/YOLOv3/)中，我们已经讲解了YOLOv3的原理。这篇，我们来折腾一下YOLOv3的训练实操。此篇中，我们不打算动手写YOLOv3的算法编程实现，而是直接使用官方代码进行编译，这样我们的折腾点都在数据集配置，调参等方面。

首先进入YOLO的官网https://pjreddie.com/darknet/yolo/ 。
YOLO从v1到v3都是支持的，但是默认是最新的v3。如果需要去到老版本，可以键入
https://pjreddie.com/darknet/yolov1/ 
或者
https://pjreddie.com/darknet/yolov2 

官网已经有比较详细操作过程，我只抽取其中一部分进行闭坑式讲解。

# 代码下载

```
$ git clone https://github.com/pjreddie/darknet
```
我们实际下载的是darknet而非YOLOv3本。因为YOLOv3只是基于darknet这个backbone而构建的应用于目标检测的专用网络，而基于darknet构建的网络很多，YOLOv3只是其中之一。比如RNN和darknet的结合可以进入：
https://pjreddie.com/darknet/rnns-in-darknet/
它实际上也是要下载darknet。

# 代码编译

```
$ cd darknet
$ make
```
这样就完成了编译，获得名为darknet的binary。
然而默认编译出来的binary是跑在CPU上的（也就是说不能通过GPU和CUDA来并行计算），且不支持OpenCV（也就是说不具备摄像头读图等功能）

想要获取以上两种功能（特别是能让你训练极度加速的GPU），你需要先改动Makefile再进行make:
```
$ vim Makefile
# 改动内容为：
    GPU=1
    ...
    OPENCV=1
    ...
$ make
```
GPU=1意味着使用Nvidia的卡及其CUDA。所以必须保证你的系统有卡，且安装了CUDA。CUDA的下载安装过程这里不赘述
OPENCV=1也需要你的系统安装了OpenCV库,，它不需要额外的硬件支持
当然，强烈建议使用GPU进行训练，它能让你的训练速度成百倍地增长。事实上，没有GPU，进入CV领域是没有意义的，你几乎做不了任何事情，除非你愿意等CPU一个月的训练时间。

# 初次测试
为了验证我们编译好的binary可用，先拿YOLOv3提供方训练好的权重进行预测
首先下载训练好的权重文件：
```
$ wget https://pjreddie.com/media/files/yolov3.weights
```
是的，光这个yolov3.weights权重文件就有237M大小。

接着，对自带的图片dog.jpg进行目标检测：
```
$ ./darknet detector test cfg/coco.data cfg/yolov3.cfg yolov3.weights data/dog.jpg
或者
$ ./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg
```
最终的检测结果会生成在predictions.jpg里，打开看看：
![](/uploads/yolov3_train_1.png)

我们再回过头来看上面的命令。为什么两条命令中的参数<font color="red"> detector test cfg/coco.data</font> 和 <font color="brown"> detect</font>是一样的效果呢？

我猜测后者是前者的封装，于是前往源代码寻找答案：
打开examples/darknet.c
```
# examples/darknet.c:

430     } else if (0 == strcmp(argv[1], "detector")){
431         run_detector(argc, argv);
432     } else if (0 == strcmp(argv[1], "detect")){
433         float thresh = find_float_arg(argc, argv, "-thresh", .5);
434         char *filename = (argc > 4) ? argv[4]: 0;
435         char *outfile = find_char_arg(argc, argv, "-out", 0);
436         int fullscreen = find_arg(argc, argv, "-fullscreen");
437         test_detector("cfg/coco.data", argv[2], argv[3], filename, thresh, .5, outfile, fullscreen);
438     }
```
可见：
当传入"detector"的时候会调用run_detector(argc, argv)          ①
当传入"detect"的时候会调用test_detector("cfg/coco.data",\*)   ②
①和②是一个效果吗？ 接下来寻找run_detector和test_detector的函数体：

打开examples/detector.c
```
# examples/detector.c:

562 void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh    , char *outfile, int fullscreen)
563 {
564     list *options = read_data_cfg(datacfg);
565     char *name_list = option_find_str(options, "names", "data/names.list");
...
...
789 void run_detector(int argc, char **argv)
790 {
...
...
836     if(0==strcmp(argv[2], "test")) test_detector(datacfg, cfg, weights, filename, thresh, hier_thresh, outfile, fu    llscreen);
```
到这里就发现，原来run_detector被传入"test"是会调用test_detector("cfg/coco.data",\*)的，所以是的①和②是一个效果

接下来还有另一个问题：有了yolov3.weights为什么还需要cfg/coco.data和cfg/yolov3.cfg呢？
因为yolov3.weights只是包含了训练生成的参数，而不含有网络信息，类别名信息等，所以我们需要引入另外的文件。
事实上，在训练和测试都需要这两个文件。
1）cfg/coco.data代表的是coco数据集的信息，我们在训练的时候需要它提供的提供“图片们存放的位置”，“类别名字有哪些”等信息，在测试的时候虽然不需要图片存放信息但是需要它给出的“类别名字有哪些”的信息，即data/names.list
2）cfg/yolov3.cfg给出的是网络的结构，训练的超参数等信息。当然训练和测试都需要它。

# 训练VOC数据集

## VOC数据集准备

首先下载VOC2007和VOC2012数据集。实际上VOC2012包含的是从2008年到2012年的数据
```
wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar xf VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar
```
数据集被统一存放在VOCdevkit/目录了，我们进入其中的2012数据集去看看：

```
$ cd VOCdevkit/VOC2012/ && ls
Annotations  ImageSets  JPEGImages  SegmentationClass  SegmentationObject
```
SegmentationClass和SegmentationObject不会被用到，因为它们是做语义分割或者实例分割用的，而YOLO是做目标检测的。
ImageSets会被用到，它存放的是哪些训练数据，哪些是验证数据的信息。
我们最关心的还是Annotations到JPEGImages的映射。举例理解：Annotations里面存放的2012_004329.xml记录了JPEGImages
里2012_004329.jpg的目标框位置，分类类别等信息。这个映射其实就是label到数据的映射。

不过可惜的是，YOLO官方代码有它自己的规则，它不接受.xml格式的label信息，它要简单的.txt格式。好在，官方提供了.xml到.txt的转换脚本，运行以下即可
```
$ python scripts/voc_label.py
```
脚本的输出包括两部分，一个是记录图片位置的.txt:
```
$ ls *txt
2007_test.txt  2007_train.txt  2007_val.txt  2012_train.txt  2012_val.txt  train.all.txt  train.txt
```
另一个就是.txt格式的label:
```
$ ls VOCdevkit/VOC2007/labels
000001.txt 000002.txt 000003.txt ...
$ ls VOCdevkit/VOC2012/labels
2008_000002.txt 2008_000003.txt ...
```
好，至此VOC的数据就准备好了。

## 开始训练VOC数据集

首先要修改cfg/voc.data文件，这样训练的时候才能知道你的要训练多少类别（classes）,训练集存放在哪(train)，验证集存放在哪(valid),类别名类别文件用哪一个(names),训练出来的中间文件和输出的权重文件存放在哪个目录(backup)
```
$ vim cfg/voc.data
  1 classes= 20
  2 train  = /home/dzi/YOLOv3/darknet/train.txt
  3 valid  = /home/dzi/YOLOv3/darknet/2007_test.txt
  4 names = data/voc.names
  5 backup = backup
```
然后要调整超参数：
```
$ vim cfg/yolov3-voc.cfg
[net]
  2 # Testing
  3 #batch=1
  4 #subdivisions=1
  5 # Training
  6 batch=64
  7 subdivisions=16
  8 width=416
  9 height=416
 10 channels=3
 11 momentum=0.9
 12 decay=0.0005
 13 angle=0
 14 saturation = 1.5
 15 exposure = 1.5
 16 hue=.1
 17 
 18 learning_rate=0.001
 19 burn_in=1000
 20 max_batches = 30000
 21 policy=steps
 22 steps=10000,20000
 23 scales=.1,.1
...
...

```
其中比较常调整是这几组参数：
1）batch=64和subdivisions=16，分别表示每次迭代用64张图片，在计算机处理的时候，每次输送16张进去。这两个参数能大则大，跟你的内存和显存有关。
2）max_batches = 30000，learning_rate=0.001， steps=10000,20000， scales=.1,.1 初始学习率为0.001，在迭代到10000次时，学习率衰减为0.001\*0.1，在迭代到20000次时，学习率继续衰减为0.001\*0.1\*0.2, 迭代到30000次停止训练。
调参策略这里就不讲了，因为太过玄学。

最后一个准备是下载用Imagenet预训练好darknet53模型。根据之前文章讲过的理论，我们知道，YOLO（以及其它目标检测网络）采用了迁移学习，也就说先用backbone网络（darknet53）在数据众多的比如Imagenet数据集上进行图片分类学习，获得强大的分类能力，然后再将网络结构接入目标检测网络部分（YOLO），继续在目标检测数据集（VOC）上进行训练。
```
$ wget https://pjreddie.com/media/files/darknet53.conv.74
```

开始正式训练!

```
$ ./darknet detector train cfg/voc.data cfg/yolov3-voc.cfg darknet53.conv.74
```

# 训练成果

用GPU训练了一天以后
...

来看一下成果：
```
$ ls -tr backup/
yolov3-voc_100.weights  yolov3-voc_400.weights  yolov3-voc_700.weights  yolov3-voc_10000.weights
yolov3-voc_200.weights  yolov3-voc_500.weights  yolov3-voc_800.weights  yolov3-voc.backup
yolov3-voc_300.weights  yolov3-voc_600.weights  yolov3-voc_900.weights

```
分别在100,200,...,900,10000次迭代之后把权重保存了下来。我们来测试一下效果

试一下100次迭代后的模型：
```
$ ./darknet detector test cfg/voc.data cfg/yolov3-voc.cfg backup/yolov3-voc_100.weights data/dog.jpg
...
chair: 65%
diningtable: 56%
person: 54%
boat: 51%
cow: 54%
motorbike: 56%
person: 54%
chair: 61%
cow: 63%
diningtable: 61%
person: 54%
bicycle: 52%
person: 53%
...
```
![](/uploads/yolov3_train_2.png)
非常惨不忍睹！根据上面打印的置信度结果来看，的确整个模型的输出结果已经错乱了

试一下900次迭代后的模型：
```
$ ./darknet detector test cfg/voc.data cfg/yolov3-voc.cfg backup/yolov3-voc_900.weights data/dog.jpg
...
data/dog.jpg: Predicted in 0.181030 seconds.
```
![](/uploads/yolov3_train_3.png)
什么变化都没有

再试一下10000次迭代后的模型：
```
$ ./darknet detector test cfg/voc.data cfg/yolov3-voc.cfg backup/yolov3-voc_10000.weights data/dog.jpg
...
data/dog.jpg: Predicted in 0.181329 seconds.
dog: 94%
car: 80%
bicycle: 92%
...
```
![](/uploads/yolov3_train_4.png)
预测结果令人满意！可以看到它的置信度给出的是dog: 94%， car: 80%，  bicycle: 92%

之后为了让计算机休息一下，我杀掉了训练进程。
不过没关系，在休息了一段时间之后，因为yolov3-voc.backup的存在，我在<font color="red">断点处</font>重启了训练：
```
./darknet detector train cfg/voc.data cfg/yolov3-voc.cfg backup/yolov3-voc.backup
```
又过了一天
...
```
ls -tr backup/
yolov3-voc_100.weights  yolov3-voc_400.weights  yolov3-voc_700.weights  yolov3-voc_10000.weights
yolov3-voc_200.weights  yolov3-voc_500.weights  yolov3-voc_800.weights  yolov3-voc_20000.weights
yolov3-voc_300.weights  yolov3-voc_600.weights  yolov3-voc_900.weights  yolov3-voc.backup
```
有了yolov3-voc_20000.weights，我想已经够了，于是主动停止了训练。

我们最后来试试效果
```
$ ./darknet detector test cfg/voc.data cfg/yolov3-voc.cfg backup/yolov3-voc_20000.weights data/dog.jpg
...
data/dog.jpg: Predicted in 0.181969 seconds.
dog: 91%
car: 89%
bicycle: 100%
...
```
![](/uploads/yolov3_train_5.png)
从置信度上看，提高显著！


以上。


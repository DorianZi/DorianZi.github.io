---
title: YOLOv2 目标检测算法之YOLOv2
date: 2019-06-18 22:09:20
tags: ["Deep Learning", "Computer Vision"]
categories: Technic
---

YOLOv2论文在[此](https://arxiv.org/pdf/1612.08242.pdf)

事实上，根据论文来看，作者根据不同的改进提出的是两个模型
1）提高YOLO的精度同时保持速度，这是YOLOv2
2）提出了一种分类和检测的联合训练策略，使得模型更强大,可以检测多达9000个类别，这是YOLO9000

我们先来看YOLOv2对于YOLOv1的改进：

# YOLOv2改进点

## 1）Batch Normalization

Batch Normalization是一个通用的提高模型泛化能力的方法，并非YOLOv2首创。翻译过来称作“批量标准化”。

要理解这个方法，首先我们要理解机器学习/深度学习的一个前提：我们之所以可以用训练数据训练出一个模型，然后用该模型对测试数据进行预测，是因为我们假设了训练数据和测试数据的分布是一致的。

换句话说，我们训练出一个好的网络，如果它有比较好的泛化能力，说明这个网络除了具有强大推理能力，还具有“保持通过它的数据分布不变”的能力。所以说一个网络被训练好，自然而然会具备这种能力，因为那些权重，偏置等参数被调整成了相应的样子。

不过，这样的话，那些权重，偏置等参数因为既要负责推理还要负责保持分布，渐渐地它们需要努力很久才能调整好。这就是我们所说的收敛速度变慢。

为了不让调整过程变慢，我们可以**主动一点，明确一点**：在网络的每一层后面（比如在CNN每个layer后面）加上专门用来保持数据分布不变的layer，这样原先那些权重，偏置参数们只需要专心训练推理能力就好了。老layer和新layer各司其职，收敛迅速，皆大欢喜。

我们主动加的layer就是Batch Normalization。

### 数学上理解

上面是定性描述，那么如何在数学上理解呢？

先看均值为0方差为1的正态分布：
![](/uploads/yolov2_1.png)
图中可看出任何一个符合该分布的数据，它有65%的概率取值在[-1,1]，95%在[-2,2]

再来看Sigmoid激活函数：
![](/uploads/yolov2_2.png)
可以看到函数的梯度在[-2,2]之外渐渐平缓，到[-4,4]之外，梯度近乎消失。

我们当然不希望数据在经过Sigmoid激活之后，梯度消失，导致训练的收敛速度变慢甚至停滞。我们希望数据经过Sigmoid激活之后还能保持不错的梯度，这样梯度下降能更快速。办法很明显：确保数据在经过Sigmoid之前，满足均值为0方差为1的正态分布。如下图所示

![](/uploads/yolov2_3.png)

### Tradeoff

上面的做法可能会引起异议：如果使用Sigmoid的[-4,4]段，甚至[-2,2]段，就相当于在用一个线性函数，而没有用到非线性部分。那激活函数还有什么意义呢？一个充满线性layer的深度网络等于没有深度，因为多个线性layer叠加等于一个线性layer。

于是我们需要对收敛速度和非线性能力做一个tradeoff(事实上这是各类算法经常遇到的，比如速度和精度的tradeoff)。方法是：把数据变换成标准正态分布之后再加一个线性偏移：y=scale\*x+shift，这样变成不太规则的正态分布，再通往激活函数。这样既能避免收敛太慢，也能使用到激活函数的非线性部分。

另外说明一下，上面新加的两个参数scale和shift的值在每个神经元上不一定相同，且也是通过训练在进行调整的。

## 2）High Resolution

我们知道，YOLOv1（以及大多数目前检测算法）为了获得强大的分类能力，在ImageNet数据集上进行了预训练，而ImageNet上的图片尺寸是224\*224，所以预训练出来的模型是针对224\*224分辨率的，而接下来进行Fine-tune时，使用的图片尺寸是448\*448，如下图：
![](/uploads/yolov2_4.png)
这样会导致Fine-tune的过程边长，因为它需要去适应新的高分辨率。

YOLOv2的解决办法如下图：
![](/uploads/yolov2_5.png)
简单说就是在ImageNet数据集（224\*224）上预训练Darknet-19网络之后，将ImageNet的数据resize为448\*448，再次训练Darknet-19网络，然后才进行Fine-tune（在Darknet-19后面接上属于YOLOv2的检测layers），并使用448\*448的检测数据集进行训练。这样就不会有分辨率适应的问题了。

## 3）Anchor Boxes

YOLOv1中经过一些列卷积layer之后，最后通过全连接层进行预测。然后reshape为7\*7\*30的特征图只是方便进行预测结果向量和原图位置的mapping. 总之7\*7\*30数据中的“30”由5,5,20三段构成。第一段的5表示第一个box的4个位置信息以及1个有无物体置信度，第二段的5表示第一个box的4个位置信息以及1个有无物体置信度，第三段的20表示两个box**共享**的20种预测类别打分。

YOLOv2去掉了信息损失多的全连接层和reshape，在特征图还是13\*13的时候，借鉴Faster RCNN里的[RPN](https://dorianzi.github.io/2019/05/26/RCNN-Series/#RPN)，即对每一个3\*3的窗口选取多个（比如9个）Anchor Box，对每个Anchor Box**单独**进行分类和位置预测。如果是9个Anchor Box，则最后得到的数据的shape是13\*13\*225，其中“225”由25,25,...,25九段构成，每一段就是每个Anchor Box的4个位置+1个置信度+20个分类打分。

前两个Box的比较见下图：
![](/uploads/yolov2_6.png)

解释一下为什么特征图为13\*13：首先YOLOv1中使用448\*448输入，YOLOv2中将其变为了416\*416然后卷积通过步长为32的卷积运算之后变成了13\*13。 这么做是因为奇数维度使得特征图只有一个中心。对于一些大物体，中心点往往落入图片中心，此时使用特征图的一个中心点去预测这些物体的边界框相对容易些。

总之显而易见，YOLOv2因为网格划分更多，且每个网格如果加入了9个Anchor Box，则总数增加为13\*13\*9=1521个Boxes，而YOLOv1中仅为7\*7\*2=98个Boxes。尽管如此YOLOv2这一改动提高了精度。这是精度换速度的tradeoff。

### 3.1) Dimension Clusters

在Anchor Box上面，YOLOv2没有简单借鉴Faster RCNN，而是进行了改进。

因为Faster RCNN里使用的Anchor Box的高宽是手动设置的先验框，所以不一定能够很好地符合Ground Truth（即IOU够高），训练速度会更慢。为了使得一开始的Anchor Box与Ground Truth的IOU够高，作者想到**事先**（在正式进行YOLOv2检测网络进行训练之前）对训练集里的所有Ground Truth进行了聚类。

聚类方法使用的是[K-means](https://dorianzi.github.io/2019/04/20/K-means/)，距离指标就是box与聚类中心box之间的IOU值：

d(box,centroid) = 1-IOU(box,centroid)

经过作者在COCO和VOC数据集上的实验，最终选取<font color="green" size="4">5个</font>聚类中心作为先验框。

### 3.2) Direct location prediction

在Anchor Box上借鉴RPN，但框位置的预测算法没有借鉴它。为什么？先看看RPN是怎么进行框位置的预测的——
首先要明白，RPN预测的框位置信息是相对Anchor box的位置偏移，该偏移表示为<img src="https://latex.codecogs.com/gif.latex?(t_x,t_y)" title="(t_x,t_y)" />

又已知Anchor box的宽和高表示为<img src="https://latex.codecogs.com/gif.latex?(w_a,h_a)" title="(w_a,h_a)" />, Anchor box的中心坐标为<img src="https://latex.codecogs.com/gif.latex?(x_a,y_a)" title="(x_a,y_a)" />，则预测框的中心<img src="https://latex.codecogs.com/gif.latex?(x,y)" title="(x,y)" />为：
<img src="https://latex.codecogs.com/gif.latex?x=(t_x\times&space;w_a)-x_a" title="x=(t_x\times w_a)-x_a" />
<img src="https://latex.codecogs.com/gif.latex?y=(t_y\times&space;w_a)-y_a" title="y=(t_y\times w_a)-y_a" />

首先我们看<img src="https://latex.codecogs.com/gif.latex?t_x" title="t_x" />或者<img src="https://latex.codecogs.com/gif.latex?t_y" title="t_y" />取值的轻微变化可以导致结果的较大变化，比如：
当<img src="https://latex.codecogs.com/gif.latex?t_x=1" title="t_x=1" />时，预测框会向右偏移一个Anchor box的宽度
当<img src="https://latex.codecogs.com/gif.latex?t_x=-1" title="t_x=-1" />时，预测框会向左偏移一个Anchor box的宽度。

然后发现上面的公式**没有任何约束**，也就是说<img src="https://latex.codecogs.com/gif.latex?t_x" title="t_x" />或者<img src="https://latex.codecogs.com/gif.latex?t_y" title="t_y" />随意取值，会导致预测框落在图片任何一个位置。这样会导致模型训练的收敛速度变慢。

考虑到上面描述的RPN预测框位置的弊端，YOLOv2决定在预测框位置这一项上**不沿用**RPN，而保留YOLOv1的做法：预测框的中心点相对于当前**网格左上角**（注意是相对于网格，而非相对于Anchor box）的偏移。偏移表示为<img src="https://latex.codecogs.com/gif.latex?t_x,t_y" title="t_x,t_y" />。为了将边界框中心点约束在当前网格中，使用sigmoid处理偏移值，这样预测的偏移值在(0,1)范围内（每个网格的宽或高分别标准化为1），用<img src="https://latex.codecogs.com/gif.latex?\sigma" title="\sigma" />表示sigmoid，则现在偏移表示为<img src="https://latex.codecogs.com/gif.latex?\sigma&space;(t_x),\sigma&space;(t_y)" title="\sigma (t_x),\sigma (t_y)" />

不过YOLOv2对框的宽度和高度的预测还是相对Anchor box的比率，而并非相对网格。预测框的宽和高相对Anchor box的比率为<img src="https://latex.codecogs.com/gif.latex?t_w,t_h" title="t_w,t_h" />，使用指数约束之后，则为<img src="https://latex.codecogs.com/gif.latex?e^{t_w},e^{t_h}" title="e^{t_w},e^{t_h}" />

设网格左上角坐标为<img src="https://latex.codecogs.com/gif.latex?(c_x,c_y)" title="(c_x,c_y)" />，Anchor box的宽和高为<img src="https://latex.codecogs.com/gif.latex?p_w,p_h" title="p_w,p_h" />，则可以得到预测框的中心坐标<img src="https://latex.codecogs.com/gif.latex?(b_x,b_y)" title="(b_x,b_y)" />以及宽高长度<img src="https://latex.codecogs.com/gif.latex?b_w,b_h" title="b_w,b_h" />分别为：

<img src="https://latex.codecogs.com/gif.latex?b_x=\sigma&space;(t_x)&plus;c_x" title="b_x=\sigma (t_x)+c_x" />
<img src="https://latex.codecogs.com/gif.latex?b_y=\sigma&space;(t_y)&plus;c_y" title="b_y=\sigma (t_y)+c_y" />
<img src="https://latex.codecogs.com/gif.latex?b_w=p_we^{t_w}" title="b_w=p_we^{t_w}" />
<img src="https://latex.codecogs.com/gif.latex?b_h=p_he^{t_h}" title="b_h=p_he^{t_h}" />

为了更好地理解上面的解释，可以再看看下图(图中<font color="blue">蓝色</font>框为预测框，虚线框为Anchor box）：
![](/uploads/yolov2_7.png)

进一步，我们知道当前的特征图大小为<img src="https://latex.codecogs.com/gif.latex?(W,H)" title="(W,H)" /> （作者模型中是(13,13)），那么我们可以将上面的位置信息表示为相对于**整张图片**的位置：

<img src="https://latex.codecogs.com/gif.latex?b_x=(\sigma&space;(t_x)&plus;c_x)/W" title="b_x=(\sigma (t_x)+c_x)/W" />
<img src="https://latex.codecogs.com/gif.latex?b_y=(\sigma&space;(t_y)&plus;c_y)/W" title="b_y=(\sigma (t_y)+c_y)/W" />
<img src="https://latex.codecogs.com/gif.latex?b_w=p_we^{t_w}/W" title="b_w=p_we^{t_w}/W" />
<img src="https://latex.codecogs.com/gif.latex?b_h=p_he^{t_h}/H" title="b_h=p_he^{t_h}/H" />

这样，我们之后要获取一个预测框的最终的位置数据，只要用上面的值乘上图片的尺寸就好了。

## 4) New Network: Darknet-19

YOLOv2使用了一个新的基础模型Darknet-19（YOLOv1用的是VGG16），它的结构从上到下如下：
![](/uploads/yolov2_8.png)

它主要采用了3\*3的卷积和2\*2的maxpooling。我们知道2\*2的maxpooling之后，特征图的维度会减少一半，但是这里设置了2倍的输出通道数，所以持平了。还可以看到每两个3\*3的卷积中间还会夹一个1\*1的卷积。

1\*1卷积核的卷积有什么用呢？

首先可以去[手写数字识别](https://dorianzi.github.io/2019/05/01/Digit-Recognizer-By-CNN/#tf-nn-conv2d%E5%87%BD%E6%95%B0)一文里回顾一下多通道卷积的实施流程。把那张图再贴过来：
![](/uploads/yolov2_10.png)
上图有两个知识点值得注意:
1. 上图中的“求和”意味着实现了原数据多个通道的整合
2. 如果你输入一个**4个**通道的数据，想要通过卷积输出**3个**通道，那么就要使用**12个**通道的卷积核。

举一反三，32通道的6\*6矩阵，用32通道的1\*1卷积核做卷积，得到的就是1通道的6x6的矩阵，如下图：
![](/uploads/yolov2_9.png)

再举一反三，192通道的28\*28矩阵，连续被**32个**192通道的1\*1卷积核做卷积，就得到**32通道**的28\*28的矩阵。其实就是所谓的信道压缩或者信道降维。如下图：
![](/uploads/yolov2_11.png)

插一句，很多时候我们不会将通道单独拎出来说，而是放入数据shape中表达。比如上面的例子就可以表达为：28\*28\*192矩阵，连续被32个1\*1\*192卷积核做卷积，得到28\*28\*32的矩阵。

所以1\*1卷积核的卷积的作用是：
1）跨通道的特征整合
2）特征通道的升维和降维
3）减少卷积核参数（简化模型）

好了，回到Darknet-19。每两个3\*3的卷积中间还会夹一个1\*1的卷积，为了就是上述的3个优化点。

最后值得一提的是，Darknet-19每个卷积层后面同样使用了Batch Normalization层以加快收敛速度。

从效果上说，Darknet-19之后，YOLOv2的mAP值没有显著提升，但是计算量却可以减少约33%，这是YOLOv2作者的实验结论。

## 5) Fine-Grained Features

YOLOv2最后输出的特征图是13\*13，对于预测大物体来说是足够的，但是对于小物体来了说这个分辨率太低了，所以YOLOv2在这里像SSD一样进行了多多尺度的检测。

不过检测方法和SSD不一样。YOLOv2提出了一种称作passthrough的layer来进行多尺度处理。从26\*26的特征图（也就说最后一个maxpooling之前）开始，**拉出一个平行分支**（原来Darknet-19网络没有被改变），进行passthrough, 举个4\*4进行passthrough的例子，如下：
![](/uploads/yolov2_12.png)
passthrough也就是从原特征图中抽取元素，输出尺寸为1/2倍，通道数为4倍的特征图。注意，不是多个新特征图，只是通道数变成了4倍。也就是说，上图中4\*4\*1特征图经过passthrough之后变成了2\*2\*4特征图，而非4个2\*2\*1特征图。

对于YOLOv2来说就是26\*26\*512通过passthrough之后，变成了13\*13\*2048 （尺寸为1/2,通道数为4倍）。

到这里passthrough分支就结束，接下来将它（13\*13\*2048的特征）与回到主网络Darknet-19（主网络经过最后一个maxpooling已经得到了13\*13\*1024）汇合，于是输出最终的13\*13\*3072特征图。在这个特征图上，进行预测操作。

## 6) Multi-Scale Training

在进行完多尺度检测之后，作者又开始在输入图片上打起了多尺度的主意。值得一提的是，这个方法是YOLOv2的创新，而不像之前讲的那些改进都是借鉴别的算法。该方法就是，在**训练过程中**，每迭代一定的次数，随机换一种尺寸（出于计算需要，尺寸都是32的倍数）的图片输入进行训练，当然这时候最后进行预测的layer也需要改，以适应这个新的尺寸。如下图所示：
![](/uploads/yolov2_13.png)
使用了该方法之后，YOLOv2可以适应不同大小的图片，并且预测出很好的结果。

# YOLO9000

YOLO9000实际上是YOLOv2的一种继续优化方法，它之所以被单独拎出来，甚至还专门命名为YOLO9000，是因为它开创性地提出了两种不同数据集联合训练方法。

ImageNet分类数据集，数据量巨大，因为图片分类是容易进行标注的；COCO，VOC等物体检测数据集，数据量较少，因为图片中物体检测的标注更复杂。YOLOv2用COCO的数据集，学习物体的边界框，置信度和框内物体分类。虽然在边界框和置信度上，没有可以加强的地方，但是在框内物体分类这一项，可以联合ImageNet分类数据集的大量数据进行分类能力的大大加强(可以识别多达9000个类别的物体)。

联合训练的最大问题是检测数据集（如COCO数据集）只有粗尺度的标记信息，如猫、狗，而分类数据集（如ImageNet）标记信息则更细尺度，如狗这一类就包括哈士奇、金毛等。所以我们需要一种一致性的方法来融合标记信息。

为此作者提出了一种名为WordTree的层级分类方法，主要思路是根据各个类别之间的从属关系（根据WordNet）建立一种树结构，如下图：
![](/uploads/yolov2_13.png)

WordTree中的根节点为"physical object"，其实就是YOLOv2给出的“框内有物体”的置信度。每个节点分出的子节点（如图中的blplane,jet,airbus,stealth fighter）都属于同一子类，可以对它们进行softmax处理，以得到每一个具体分类的概率。当然上面经过softmax得到的概率只是局部概率，要计算一个分类的最终预测概率，需要将从根节点到该分类的路径上的各个局部概率相乘。如计算jet的最终预测概率，需要将physical object,artifact,vehicle,air,airplane,jet这6个节点的局部概率相乘。

这个结构在训练上的思路是：在训练过程中，当网络遇到一个检测数据集的图片与标记信息，那么就把这些数据用完整的损失函数反向传播；当网络遇到一个分类数据集的图片和分类标记信息，只用整个结构中分类部分的损失函数反向传播。

在预测上，当然跟所有软分类器一样，计算所有路径的概率乘积，选取最大概率的那个结果，就是最终的预测结果了。

以上。

# 参考

https://zhuanlan.zhihu.com/p/35325884
https://www.cnblogs.com/guoyaohua/p/8724433.html
http://lanbing510.info/2017/09/04/YOLOV2.html

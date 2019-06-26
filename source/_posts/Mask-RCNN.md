---
title: Mask RCNN 用于实例分割的Mask RCNN网络
date: 2019-06-05 11:29:02
tags: ["Deep Learning", "Computer Vision"]
categories: Technic
---

论文在[此](https://arxiv.org/pdf/1703.06870.pdf)

Mask RCNN是在[Faster RCNN](https://dorianzi.github.io/2019/05/26/RCNN-Series/#Faster-RCNN)基础上的改进算法。这里之所以单独成文是因为Mask RCNN不仅仅用于目标检测，还用于实例分割。

目标检测和实例分割的区别在于，实例分割不仅仅需要将目标识别，还需要将它的轮廓绘出。这意味着需要对每一个像素进行分类。

![](/uploads/RCNN_1.png)

# Mask RCNN的改进点

先看Mask RCNN的流程图：

![](/uploads/mask_rcnn_1.png)

另一种表示：

![](/uploads/mask_rcnn_2.png)

1）将ROI Pooling改进为ROI Align
2) 设计与分类和box回归网络平行的第三个分支：基于FCN的mask层用于图像分割

## ROI Align

ROI Algin的引入是为了解决ROI Pooling精度上的缺陷
ROI Pooling在操作过程中，因为使用了量化，导致proposal框被“四舍五入”，与原框有了误差：

![](/uploads/mask_rcnn_3.png)

误差来自于两个阶段：

1） 当从feature map上去圈出原图中的框映射过来的ROI时，因为不能保证原图的框坐标正好可以整除缩小到当前feature map上，所以会有量化误差（注意图中黑线边框）：

![](/uploads/mask_rcnn_4.png)

2）接着，需要进行Max Pooling过程。在[前文的章节](https://dorianzi.github.io/2019/05/26/RCNN-Series/#ROI-Pooling-Layer)讲过，为了保持输出结果的shape一致，可以通过高宽分别划分来进行。

现在上图为8\*9，需要Max Pooling为3\*3，很显然宽可以划分，高被划分3段会出现浮点数。为了避免浮点数，需要将高量化为最近的6以便整除3，所以就是6\*9：

![](/uploads/mask_rcnn_5.png)

这样经过两个误差之后，ROI其实跟原图所希望圈出的目标框有了不小的误差。

为了避免这个误差，ROI Align决定取消两次量化过程，让浮点数留下来。这样的话，高被分为8/3的三段，宽是3的3段。一共9个区域，每个区域的大小为(8/3,3)，如上上图的红线部分。

接下来，进行Max Pooling。此时发现9个区域中的每一个区域都有被“斩断”的像素，我们不知道它的值，因此无法找出哪个才是Max的。怎么办？

此时ROI Align中提出的方法是采用双线性插值来计算虚拟像素点的值，然后进行Max Pooling

![](/uploads/mask_rcnn_6.png)

如图所示，将9个区域里的每一个区域都等分成4份，并且取中心点值，该值由[双线性插值](https://baike.baidu.com/item/%E5%8F%8C%E7%BA%BF%E6%80%A7%E6%8F%92%E5%80%BC/11055945?fr=aladdin)方法获得。然后将这4个点进行Max Pooling，获得代表该区域的值。

## 基于FCN的Mask层

FCN即Fully Convolutional Networks，特点是输出和输入的shape一致。这个特性意味着一张图片在经过FCN之后能保留每一个像素的一对一映射，这是非常适合用来进行图像分割的：

![](/uploads/mask_rcnn_7.png)

在网络结构上是这样的：

1）将常规CNN网络的最后3个全连接层换成了3个卷积层，这3个层的卷积核大小分别为(1,1,4096),(1,1,4096)和(1,1,21)。意味着<font color=Green>每个ROI</font>经过这3个层之后，数据变成了一个21维向量，或者说通道数为21的1\*1矩阵。21维向量的每一个数值可以理解为21种类别中某一个的概率值。

2）将21个通道的数据通过反卷积转换为与原图shape一致的特征图，实现像素级的分类

下图给出了每个ROI和全图的概率转换（种类为1000种）

![](/uploads/mask_rcnn_8.png)

上面提到了一个关键的技术：反卷积

### 反卷积

反卷积(Deconvolution)容易导致误会，叫“转置卷积(Transposed Convolution)”更加合适。

卷积起到降维作用：

![](/uploads/mask_rcnn_9.gif)

反卷积起到还原作用：

![](/uploads/mask_rcnn_10.gif)

也可以使用padding进行反卷积：

![](/uploads/mask_rcnn_12.gif)


这两个卷积核之间的关系可以通过代数推导：

将矩阵扁平化成为向量X,它通过卷积之后得到向量Y，则有：

<img src="https://latex.codecogs.com/gif.latex?X=CY" />

实际上C是下面的稀疏矩阵：

![](/uploads/mask_rcnn_11.png)

矩阵进行逆运算：

<img src="https://latex.codecogs.com/gif.latex?X=C^{T}Y" />

也就是说<img src="https://latex.codecogs.com/gif.latex?C^{T}" />代表反卷积所用的卷积核。

以上。

# 参考
https://blog.csdn.net/gusui7202/article/details/84799535
https://buptldy.github.io/2016/10/29/2016-10-29-deconv/




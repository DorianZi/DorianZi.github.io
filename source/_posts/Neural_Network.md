---
title: Neural Network 神经网络
date: 2019-04-22 17:15:02
tags: ["Algorithm","Deep Learning"]
categories: Technic
---

# 神经网络结构

神经网络的结构如下

![](/uploads/neural_network_1.png)

我们只需要从这个图中得到这几点：
1） 输入和输出的个数是固定的
2） 隐藏层是可变的，它的变化就是神经网络设计结构的变化
3） 连线是权重
4） 连线的交汇圈包含求和以及激活函数，是输出，也可以作为下一个隐藏层的输入
5） 圆圈的发散出的多条连线是同一个输入的多份拷贝

关于权重和激活函数，见展开说明的神经网络图：

![](/uploads/neural_network_2.png)

于是上图的代数关系是：

<img src="https://latex.codecogs.com/gif.latex?Z=sgn(a_1w_1&plus;a_2w_2&plus;a_3w_3)" title="Z=sgn(a_1w_1+a_2w_2+a_3w_3)" />

那么sgn是什么呢？根据需要，它可以原样输出，也可以是具体的激活函数函数，比如你定义的输出是概率，那么可以使用sigmoid函数,用来把量值映射到0到1之间。

插一句，在我看来，激活函数是神经网络的精髓，因为它模拟了神经元的工作方式：输入汇集了，不代表神经元会继续传播它，需要看有没有达到阈值，就算达到阈值了，传播出去的信号不一定是原本的量值，而是通过一定规则转换的。

看到求和，很容易想到之前讲过的[感知机](https://dorianzi.github.io/2019/03/29/Perceptron/)，而看到sigmoid又想到[逻辑回归](https://dorianzi.github.io/2019/03/28/Logistic-Regression/)。它们跟神经网络有什么关系呢？来探个究竟：

## 感知机和神经网络

一句话：感知机就是神经网络！

在感知机的分类中，是找到一个超平面<img src="https://latex.codecogs.com/gif.latex?WX^{T}&plus;b=0"  />，( 其中<img src="https://latex.codecogs.com/gif.latex?W=(w_{1},w_{2},...,w_{n})"  /> )来准确地将所有两类数据准确区分在超平面的两边。

也就是说一个点<img src="https://latex.codecogs.com/gif.latex?(x_{1},x_{2},...,x_n)" title="(x_1,x_2,...,x_n)" />被感知机分类，也就是判断：

<img src="https://latex.codecogs.com/gif.latex?W\begin{pmatrix}&space;x_{1}\\&space;x_{2}\\&space;...\\&space;x_{n}&space;\end{pmatrix}>-b" /> 是一类

<img src="https://latex.codecogs.com/gif.latex?W\begin{pmatrix}&space;x_{1}\\&space;x_{2}\\&space;...\\&space;x_{n}&space;\end{pmatrix}<-b" /> 是另一类

用神经网络表示如下：

我们构建一个神经网络：n个输入是<img src="https://latex.codecogs.com/gif.latex?x_1,x_2,...,x_n" title="x_1,x_2,...,x_n" />，权重分别是<img src="https://latex.codecogs.com/gif.latex?w_1,w_2,...,w_n" title="w_1,w_2,...,w_n" />，求和之后，sgn为原样输出，则：

![](/uploads/neural_network_3.png)

要求神经网络的输出：

<img src="https://latex.codecogs.com/gif.latex?Z>-b" /> 是一类

<img src="https://latex.codecogs.com/gif.latex?Z<-b" /> 是另一类

可见，感知机就是单层的神经网络（这里以连接线的层数表示网络层数）

## 逻辑回归

逻辑回归也是单层神经网络，证明如下：

构建神经网络——

![](/uploads/neural_network_4.png)

则:

<img src="https://latex.codecogs.com/gif.latex?Y=\frac{1}{1&plus;e^{-\theta_1x_1-\theta_2x_2-\theta_3x_3}}" title="Y=\frac{1}{1+e^{-\theta_1x_1-\theta_2x_2-\theta_3x_3}}" />

证毕。

# 神经网络的矩阵表示

通过矩阵相乘可以表示一个神经网络，以单层为例：

![](/uploads/neural_network_5.png)

上面的过程可以表示为：

<img src="https://latex.codecogs.com/gif.latex?[x_1,x_2,x_3]\begin{bmatrix}&space;w_{11}&space;&&space;w_{12}\\&space;w_{21}&space;&&space;w_{22}&space;\\&space;w_{31}&space;&&space;w_{32}&space;\end{bmatrix}=&space;[y_1,y_2]" title="[x_1,x_2,x_3]\begin{bmatrix} w_{11} & w_{12}\\ w_{21} & w_{22} \\ w_{31} & w_{32} \end{bmatrix}= [y_1,y_2]" />

或者取转置，为：

<img src="https://latex.codecogs.com/gif.latex?\begin{bmatrix}&space;w_{11}&space;&&space;w_{21}&space;&&space;w_{31}\\&space;w_{12}&space;&&space;w_{22}&space;&&space;w_{32}&space;\end{bmatrix}&space;\begin{bmatrix}&space;x_{1}\\&space;x_{2}\\&space;x_{3}&space;\end{bmatrix}=&space;\begin{bmatrix}&space;y_{1}\\&space;y_{2}&space;\end{bmatrix}" title="\begin{bmatrix} w_{11} & w_{21} & w_{31}\\ w_{12} & w_{22} & w_{32} \end{bmatrix} \begin{bmatrix} x_{1}\\ x_{2}\\ x_{3} \end{bmatrix}= \begin{bmatrix} y_{1}\\ y_{2} \end{bmatrix}" />

如果是多层神经网络，则为多个权重矩阵连乘

# 神经网络的偏置

![](/uploads/neural_network_6.png)

偏置可以看作对除了输出层的每一层（输入层和隐藏层）加入一个平行的常数输入：1

它的权重为<img src="https://latex.codecogs.com/gif.latex?b^{(1)},b^{(2)},...,b^{(m)}" title="b^{(1)},b^{(2)},...,b^{(m)}" />  (假设局部输出为m个)

那么对于上面的例子来说，可以用矩阵表示为：

<img src="https://latex.codecogs.com/gif.latex?\begin{bmatrix}&space;w_{11}&space;&&space;w_{21}&space;&&space;w_{31}\\&space;w_{12}&space;&&space;w_{22}&space;&&space;w_{32}&space;\end{bmatrix}&space;\begin{bmatrix}&space;x_{1}\\&space;x_{2}\\&space;x_{3}&space;\end{bmatrix}&plus;\begin{bmatrix}&space;b_{1}\\&space;b_{2}\\&space;\end{bmatrix}=&space;\begin{bmatrix}&space;y_{1}\\&space;y_{2}&space;\end{bmatrix}" title="\begin{bmatrix} w_{11} & w_{21} & w_{31}\\ w_{12} & w_{22} & w_{32} \end{bmatrix} \begin{bmatrix} x_{1}\\ x_{2}\\ x_{3} \end{bmatrix}+\begin{bmatrix} b_{1}\\ b_{2}\\ \end{bmatrix}= \begin{bmatrix} y_{1}\\ y_{2} \end{bmatrix}" />

# 神经网络的训练

训练网络就是在结构已经确定的情况下，训练权重和偏置。

首先我们从目标函数入手。输出值作为预测值，那么预测值和label值是有差别的，我们用常见的均方误差来表示这个差别：

<img src="https://latex.codecogs.com/gif.latex?e=\frac{1}{2}(a-y^{(i)})^2" title="e=\frac{1}{2}(a-y^{(i)})^2" />

其中<img src="https://latex.codecogs.com/gif.latex?a" title="a" />为预测值，它可以通过权重来表示，<img src="https://latex.codecogs.com/gif.latex?y^{(i)}" title="y^{(i)}" />为第i个样本的label值。

容易想到，我们只要将e对相应权重或偏置求偏导，然后通过梯度下降法，使得e最小，就能找到该权重或偏置的最优值。我们来推导求偏导公式，为便于理解，考虑下面的网络（假设没有激活函数或者理解激活函数为原样输出）：

![](/uploads/neural_network_7.png)

比如我们考虑第一层权重对e的影响，采用链式求导：

<img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;\frac{\partial&space;e}{\partial&space;w_{11}^{(1)}}=&space;&\frac{\partial&space;e}{\partial&space;a_{1}^{(3)}}*&space;\frac{\partial&space;a_{1}^{(3)}}{\partial&space;a_{1}^{(2)}}*&space;\frac{\partial&space;a_{1}^{(2)}}{w_{11}^{(1)}}\\=&space;&\frac{\partial&space;e}{\partial&space;a_{1}^{(3)}}*&space;\frac{\partial&space;a_{1}^{(3)}}{\partial&space;a_{1}^{(2)}}*&space;a_{1}^{(1)}&space;\end{align*}" />

<img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;\frac{\partial&space;e}{\partial&space;w_{12}^{(1)}}=&space;&\frac{\partial&space;e}{\partial&space;a_{1}^{(3)}}*&space;\frac{\partial&space;a_{1}^{(3)}}{\partial&space;a_{2}^{(2)}}*&space;\frac{\partial&space;a_{2}^{(2)}}{w_{12}^{(1)}}\\=&space;&\frac{\partial&space;e}{\partial&space;a_{1}^{(3)}}*&space;\frac{\partial&space;a_{1}^{(3)}}{\partial&space;a_{2}^{(2)}}*&space;a_{1}^{(1)}&space;\end{align*}" />

考虑第二层权重：

<img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;\frac{\partial&space;e}{\partial&space;w_{11}^{(2)}}=&space;&\frac{\partial&space;e}{\partial&space;a_{1}^{(3)}}*&space;\frac{\partial&space;a_{1}^{(3)}}{\partial&space;w_{11}^{(2)}}\\&space;=&\frac{\partial&space;e}{\partial&space;a_{1}^{(3)}}*&space;a_1^{(2)}&space;\end{align*}" title="\begin{align*} \frac{\partial e}{\partial w_{11}^{(2)}}= &\frac{\partial e}{\partial a_{1}^{(3)}}* \frac{\partial a_{1}^{(3)}}{\partial z}* \frac{\partial z}{\partial w_{11}^{(2)}}\\ =&\frac{\partial e}{\partial a_{1}^{(3)}}* \frac{\partial a_{1}^{(3)}}{\partial z}* a_1^{(2)} \end{align*}" />

<img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;\frac{\partial&space;e}{\partial&space;w_{21}^{(2)}}=&space;&\frac{\partial&space;e}{\partial&space;a_{1}^{(3)}}*&space;\frac{\partial&space;a_{1}^{(3)}}{\partial&space;w_{21}^{(2)}}\\&space;=&\frac{\partial&space;e}{\partial&space;a_{1}^{(3)}}*&space;a_2^{(2)}&space;\end{align*}" title="\begin{align*} \frac{\partial e}{\partial w_{21}^{(2)}}= &\frac{\partial e}{\partial a_{1}^{(3)}}* \frac{\partial a_{1}^{(3)}}{\partial w_{21}^{(2)}}\\ =&\frac{\partial e}{\partial a_{1}^{(3)}}* a_2^{(2)} \end{align*}" />

第一层偏置：

<img src="https://latex.codecogs.com/gif.latex?\begin{align*}\frac{\partial&space;e}{\partial&space;b_{1}^{(1)}}=&space;&\frac{\partial&space;e}{\partial&space;a_{1}^{(3)}}*&space;\frac{\partial&space;a_{1}^{(3)}}{\partial&space;a_{1}^{(2)}}*&space;\frac{\partial&space;a_{1}^{(2)}}{\partial&space;b_{1}^{(1)}}\\=&space;&\frac{\partial&space;e}{\partial&space;a_{1}^{(3)}}*&space;\frac{\partial&space;a_{1}^{(3)}}{\partial&space;a_{1}^{(2)}}\end{align*}" />

<img src="https://latex.codecogs.com/gif.latex?\begin{align*}\frac{\partial&space;e}{\partial&space;b_{2}^{(1)}}=&space;&\frac{\partial&space;e}{\partial&space;a_{1}^{(3)}}*&space;\frac{\partial&space;a_{1}^{(3)}}{\partial&space;a_{2}^{(2)}}*&space;\frac{\partial&space;a_{2}^{(2)}}{\partial&space;b_{2}^{(1)}}\\=&space;&\frac{\partial&space;e}{\partial&space;a_{1}^{(3)}}*&space;\frac{\partial&space;a_{1}^{(3)}}{\partial&space;a_{2}^{(2)}}\end{align*}" title="\begin{align*}\frac{\partial e}{\partial b_{2}^{(1)}}= &\frac{\partial e}{\partial a_{1}^{(3)}}* \frac{\partial a_{1}^{(3)}}{\partial a_{2}^{(2)}}* \frac{\partial a_{2}^{(2)}}{\partial b_{2}^{(1)}}\\= &\frac{\partial e}{\partial a_{1}^{(3)}}* \frac{\partial a_{1}^{(3)}}{\partial a_{2}^{(2)}}\end{align*}" />

第二层偏置：

<img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;\frac{\partial&space;e}{\partial&space;b_{1}^{(2)}}=&space;&\frac{\partial&space;e}{\partial&space;a_{1}^{(3)}}*&space;\frac{\partial&space;a_{1}^{(3)}}{\partial&space;b_{1}^{(2)}}\\&space;=&space;&\frac{\partial&space;e}{\partial&space;a_{1}^{(3)}}\end{align*}" title="\begin{align*} \frac{\partial e}{\partial b_{1}^{(2)}}= &\frac{\partial e}{\partial a_{1}^{(3)}}* \frac{\partial a_{1}^{(3)}}{\partial z}* \frac{\partial z}{\partial b_{1}^{(2)}}\\ = &\frac{\partial e}{\partial a_{1}^{(3)}}* \frac{\partial a_{1}^{(3)}}{\partial z} \end{align*}" />

上面任何一个公式都可以得到一个关于被求偏导变量的梯度表达式，只要采用<font size="4">梯度下降</font>就能得到迭代公式，就可以开始训练了！

不过，在梯度下降前，有没有更优化的方法呢？ 乍一看，各种权重和偏置的计算当中有一些共有的部分，你能找出准确的代数关系以方便我们不重复计算共有部分吗？如果你能找出来，那么你可能发明了<font size="4">Back Propgation反向传播算法</font>：

首先，我们人为定义一种“误差”。这种“误差”就是总损失e对各层各神经元的偏导。

比如第1,2,3层第1个神经元的误差分别为：

<img src="https://latex.codecogs.com/gif.latex?\delta&space;_1^{(1)}=\frac{\partial&space;e}{\partial&space;a_1^{(1)}}=\frac{\partial&space;e}{\partial&space;a_1^{(3)}}*\frac{\partial&space;a_1^{(3)}}{\partial&space;a_1^{(2)}}*\frac{\partial&space;a_1^{(2)}}{\partial&space;a_1^{(1)}}" title="\frac{\partial e}{\partial a_1^{(1)}}=\frac{\partial e}{\partial a_1^{(3)}}*\frac{\partial a_1^{(3)}}{\partial z}*\frac{\partial z}{\partial a_1^{(2)}}*\frac{\partial a_1^{(2)}}{\partial a_1^{(1)}}" />

<img src="https://latex.codecogs.com/gif.latex?\delta&space;_1^{(2)}=\frac{\partial&space;e}{\partial&space;a_1^{(2)}}=\frac{\partial&space;e}{\partial&space;a_1^{(3)}}*\frac{\partial&space;a_1^{(3)}}{\partial&space;a_1^{(2)}}" title="\frac{\partial e}{\partial a_1^{(2)}}=\frac{\partial e}{\partial a_1^{(3)}}*\frac{\partial a_1^{(3)}}{\partial z}*\frac{\partial z}{\partial a_1^{(2)}}" />


<img src="https://latex.codecogs.com/gif.latex?\delta&space;_1^{(3)}=\frac{\partial&space;e}{\partial&space;a_1^{(3)}}=\frac{\partial&space;e}{\partial&space;a_1^{(3)}}" title="\frac{\partial e}{\partial a_1^{(3)}}=\frac{\partial e}{\partial a_1^{(3)}}" />


可以推得误差的<font size="4">反向</font>迭代公式：

<img src="https://latex.codecogs.com/gif.latex?\delta&space;_i^{(j)}=\delta&space;_i^{(j&plus;1)}\frac{\partial&space;a_i^{(j&plus;1)}}{\partial&space;a_i^{(j)}}" title="\delta _i^{(j)}=\delta _i^{(j+1)}\frac{\partial a_i^{(j+1)}}{\partial a_i^{(j)}}" />&emsp;&emsp;①

再代入上面的权重的偏导：

<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;e}{\partial&space;w_{11}^{(1)}}=\delta&space;_1^{(2)}a_1^{(1)}" title="\frac{\partial e}{\partial w_{11}^{(1)}}=\delta _1^{(2)}a_1^{(1)}" />

<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;e}{\partial&space;w_{12}^{(1)}}=\delta&space;_2^{(2)}a_1^{(1)}" title="\frac{\partial e}{\partial w_{11}^{(1)}}=\delta _1^{(2)}a_1^{(1)}" />

<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;e}{\partial&space;w_{11}^{(2)}}=\delta&space;_1^{(3)}a_1^{(2)}" title="\frac{\partial e}{\partial w_{11}^{(1)}}=\delta _1^{(2)}a_1^{(1)}" />

<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;e}{\partial&space;w_{21}^{(2)}}=\delta&space;_1^{(3)}a_2^{(2)}" title="\frac{\partial e}{\partial w_{11}^{(1)}}=\delta _1^{(2)}a_1^{(1)}" />

推得权重的反向迭代公式：

<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;e}{\partial&space;w_{ki}^{(j)}}=\delta&space;_i^{(j+1)}a_k^{(j)}" title="\frac{\partial e}{\partial w_{11}^{(1)}}=\delta _1^{(2)}a_1^{(1)}" />&emsp;&emsp;②

同理对于偏置的偏导：

<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;e}{\partial&space;b_{1}^{(1)}}=\delta&space;_1^{(2)}" title="\frac{\partial e}{\partial b_{1}^{(1)}}=\delta _1^{(2)}" />

<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;e}{\partial&space;b_{2}^{(1)}}=\delta&space;_2^{(2)}" title="\frac{\partial e}{\partial b_{1}^{(1)}}=\delta _1^{(2)}" />

<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;e}{\partial&space;b_{1}^{(2)}}=\delta&space;_1^{(3)}" title="\frac{\partial e}{\partial b_{1}^{(1)}}=\delta _1^{(2)}" />

推得偏置的反向迭代公式：

<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;e}{\partial&space;b_{i}^{(j)}}=\delta&space;_i^{(j+1)}" title="\frac{\partial e}{\partial b_{1}^{(1)}}=\delta _1^{(2)}" />&emsp;&emsp;③

通过上面的迭代公式，我们可以更快速地计算出偏导数。求出来之后？跟之前一样，采用梯度下降就好了：

<img src="https://latex.codecogs.com/gif.latex?new=old-\eta&space;\Delta" title="new=old-\eta \Delta" />

以上。

# 参考

https://blog.csdn.net/illikang/article/details/82019945
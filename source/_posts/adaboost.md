---
title: AdaBoost 提升算法
date: 2019-04-05 09:07:25
tags: ["Algorithm"]
categories: Technic
---

# 提升方法

对一个复杂任务来说，多个专家加权判断比一个专家单独判断效果要好，在机器学习中也一样。并且在机器学习中，训练一个高精度的模型比训练多个稍微粗糙的模型要困难得多，于是boosting(提升)方法被提出了。

提升方法在训练中主要需要解决两个问题：

1）数据的权重：基于什么原则向训练数据赋予权重；

2）分类器的权重：多个不同的弱分类器如何结合成为一个强分类器

AdaBoost很好地回答了上面两个问题。

# AdaBoost提升方法

AdaBoost对上面两个问题的回答是：

1）数据的权重：在本轮训练中分类错误的数据被赋予更高的权重，以便在下一轮训练中被重点关照

2）分类器的权重：根据误差率赋予弱分类器权重，然后采用加权表决的方法获得强分类器

## 通过例子理解AdaBoost

有以下样本集，x为特征，y为分类值，共有-1,1两类，要求用AdaBoost训练一个强分类器

![](/uploads/adaboost_1.png)

<font color="Blue" size="4">第1轮训练</font>
1）初始化训练数据的权重：

<img src="https://latex.codecogs.com/gif.latex?D_1=(w_{11},w_{12},...,w_{110})" title="D_1=(w_{11},w_{12},...,w_{110})" />

其中<img src="https://latex.codecogs.com/gif.latex?w_{1i}=0.1,\&space;\&space;i=1,2,...,10" title="w_{1i}=0.1,\ \ i=1,2,...,10" />

2）基于<img src="https://latex.codecogs.com/gif.latex?D_1" title="D_1" />，计算<font color="Red">特征x</font>的哪个阈值使得分类误差最低。注意，一个阈值需要计算正向和反向<font color="Red">两种情况</font>：

![](/uploads/adaboost_2.png)

注意，计算时应乘以权重。

可见，当特征x的阈值为2.5或8.5时，误差率最低，为0.3，选择其中任何一个都可以，**本轮选择v=2.5**构建弱分类器：

<img src="https://latex.codecogs.com/gif.latex?G_1(x)=\left\{\begin{matrix}&space;1,\&space;\&space;\&space;\&space;x<2.5\\&space;-1,\&space;\&space;x>2.5&space;\end{matrix}\right." />

3) <img src="https://latex.codecogs.com/gif.latex?G_1(x)" title="G_1(x)" />在训练数据集上的误差率为:

<img src="https://latex.codecogs.com/gif.latex?e_1=P(G_1(x_i)\neq&space;y_i)=0.3" title="e_1=P(G_1(x_i)\neq y_i)=0.3" />

4）通过误差率计算弱分类器的权重系数：

<img src="https://latex.codecogs.com/gif.latex?\alpha_1=\frac{1}{2}ln\frac{1-e_1}{e_1}=0.4236"  />

5) 通过弱分类器加权获得强分类器：

<img src="https://latex.codecogs.com/gif.latex?f_1(x)=0.4236G_1(x)" title="f_1(x)=0.4236G_1(x)" />

加上信号函数才变成我们要的分类器决策函数：

<img src="https://latex.codecogs.com/gif.latex?C(x)=sign[f_1(x)]" title="C(x)=sign[f_1(x)]" />

6）测试一下强分类器对样本数据的分类能力：

![](/uploads/adaboost_3.png)

有3个误分点，还需要继续boost

7）因为需要继续boost,根据本轮生成的弱分类器的错误率在强分类器里的权重，重新计算各数据的权重，供下轮训练使用：

更数据权重的公式为：

<img src="https://latex.codecogs.com/gif.latex?w_{mi}=\frac{w_{(m-1)i}e^{-\alpha_{m-1}y_iG_{m-1}(x_i)}}{\sum_{i=1}^{10}w_{(m-1)i}e^{-\alpha_{m-1}y_iG_{m-1}(x_i)}}" title="w_{mi}=\frac{w_{(m-1)i}e^{-\alpha_{m-1}y_iG_{m-1}(x_i)}}{\sum_{i=1}^{10}w_{(m-1)i}e^{-\alpha_{m-1}y_iG_{m-1}(x_i)}}" />

m表示下一轮是第m轮训练，i表示第i个数据。研究一下上面的公式，<img src="https://latex.codecogs.com/gif.latex?y_iG_{m-1}(x_i)"  />的正负代表该点（第i点）是否分类正确，所以它是用来当前点是被分对还是分错的，然后乘上本轮若分类器权重：<img src="https://latex.codecogs.com/gif.latex?\alpha_{m-1}y_iG_{m-1}(x_i)"  />，就相当于在全局强分类器的角度看待当前点分类的对错。然后用自然对数底e作幂来获得当前数据权重更新的信息。至于为什么用自然对数，是基于数学家们的验证，这里不作证明。

据上公式，求第2轮的所有数据的权重:

<img src="https://latex.codecogs.com/gif.latex?D_2=(w_{21},w_{22},...,w_{210})\\=(0.07143,0.07143,0.07143,0.07143,0.07143,0.07143,0.16667,0.16667,0.16667,0.07143)" title="D_2=(w_{21},w_{22},...,w_{210})" />

<font color="Blue" size="4">第2轮训练</font>
8）相当于回到2）开始循环，基于<img src="https://latex.codecogs.com/gif.latex?D_2"  />，计算特征x的哪个阈值使得分类误差最低：

![](/uploads/adaboost_4.png)

可见在8.5时候误分率最低，为0.21429，则**本轮选择v=8.5**构建弱分类器：

<img src="https://latex.codecogs.com/gif.latex?G_2(x)=\left\{\begin{matrix}&space;1,\&space;\&space;\&space;\&space;x<8.5\\&space;-1,\&space;\&space;x>8.5&space;\end{matrix}\right." />

9) <img src="https://latex.codecogs.com/gif.latex?G_2(x)"  />在训练数据集上的误差率为:

<img src="https://latex.codecogs.com/gif.latex?e_2=P(G_2(x_i)\neq&space;y_i)=0.21429" title="e_1=P(G_1(x_i)\neq y_i)=0.3" />

10）通过误差率计算弱分类器的权重系数：

<img src="https://latex.codecogs.com/gif.latex?\alpha_2=\frac{1}{2}ln\frac{1-e_2}{e_2}=0.6496"  />

11) 通过弱分类器加权获得强分类器：

<img src="https://latex.codecogs.com/gif.latex?f_2(x)=0.4236G_1(x)+0.6496G_2(x)"  />

加上信号函数才变成我们要的分类器决策函数：

<img src="https://latex.codecogs.com/gif.latex?C(x)=sign[f_2(x)]"  />

12）测试一下强分类器对样本数据的分类能力：

![](/uploads/adaboost_5.png)

还有3个误分点，还需要继续boost

13）因为需要继续boost,根据本轮生成的弱分类器的错误率在强分类器里的权重，重新计算各数据的权重，供下轮训练使用：

根据<img src="https://latex.codecogs.com/gif.latex?w_{3i}=\frac{w_{2i}e^{-\alpha_2y_iG_2(x_i)}}{\sum_{i=1}^{10}w_{2i}e^{-\alpha_2y_iG_2(x_i)}}"  />，求第3轮的所有数据的权重:

<img src="https://latex.codecogs.com/gif.latex?D_3=(w_{31},w_{32},...,w_{310})\\=(0.0455,0.0455,0.0455,0.1667,0.1667,0.1667,0.1060,0.1060,0.1060,0.0455)"/>

<font color="Blue" size="4">第3轮训练</font>
14）相当于再次回到2）开始循环，基于<img src="https://latex.codecogs.com/gif.latex?D_3"  />，计算特征x的哪个阈值使得分类误差最低：

![](/uploads/adaboost_6.png)

可见在5.5时候误分率最低，为0.1819，则**本轮选择v=5.5**构建弱分类器：

<img src="https://latex.codecogs.com/gif.latex?G_3(x)=\left\{\begin{matrix}&space;-1,\&space;\&space;x<5.5\\&space;1,\&space;\&space;\&space;\&space;x>5.5&space;\end{matrix}\right." />

15) <img src="https://latex.codecogs.com/gif.latex?G_3(x)"  />在训练数据集上的误差率为:

<img src="https://latex.codecogs.com/gif.latex?e_3=P(G_3(x_i)\neq&space;y_i)=0.1819"  />

16）通过误差率计算弱分类器的权重系数：

<img src="https://latex.codecogs.com/gif.latex?\alpha_3=\frac{1}{2}ln\frac{1-e_3}{e_3}=0.7514"  />

17) 通过弱分类器加权获得强分类器：

<img src="https://latex.codecogs.com/gif.latex?f_3(x)=0.4236G_1(x)+0.6496G_2(x)+0.7514G_3(x)"  />

加上信号函数才变成我们要的分类器决策函数：

<img src="https://latex.codecogs.com/gif.latex?C(x)=sign[f_3(x)]"  />

18）测试一下强分类器对样本数据的分类能力：

![](/uploads/adaboost_7.png)

有0个误分点，boost结束！

以上。

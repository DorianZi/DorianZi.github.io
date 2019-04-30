---
title: Logistic Regression Classifier 逻辑回归分类器
date: 2019-03-28 14:02:15
tags: ["Algorithm"]
categories: ["Technic"]
---

## 问题的提出

我们考虑一个股票推荐的数据集，它每行是一个样本。每个样本有四个特征，推荐结果只有“推荐”与“不推荐”两类：

![](/uploads/logistic_regression_pic_1.png)

基于以上的数据集，我们训练一个分类器。首先将推荐结果量化表示：1表示推荐，0表示不推荐。

那么对前3个样本<img src="https://latex.codecogs.com/gif.latex?X1,X2,X3"  />中的任何一个<img src="https://latex.codecogs.com/gif.latex?X" title="X" />，我们希望分类器应该能将它分类到1，而对后3个样本<img src="https://latex.codecogs.com/gif.latex?X4,X5,X6"  />中的任何一个<img src="https://latex.codecogs.com/gif.latex?X" title="X" />，我们希望分类器应该能将它分类到0。

从这里开始，假如用线性回归的思维，我们可以训练出一个这样的模型：<img src="https://latex.codecogs.com/gif.latex?f(X)=V^{T}X+C"  />，当<img src="https://latex.codecogs.com/gif.latex?f(X)>0" title="f(X)>0" />时，归到1类；当<img src="https://latex.codecogs.com/gif.latex?f(X)>0" title="f(X)>0" />时，归到0类。 这样分类精度会很低。

我们这里要使用的是逻辑回归，也就是Logistic Regression。注意它不是回归，而它是一种分类算法。

## 逻辑回归

### 概率模型

让我们的分类器输出概率，并设定决策方式为：分类到概率更大的类别，即：对前3个样本<img src="https://latex.codecogs.com/gif.latex?X1,X2,X3"  />中的任何一个<img src="https://latex.codecogs.com/gif.latex?X" title="X" />，分类器应该将它分类到1的概率大于0.5; 对后3个样本<img src="https://latex.codecogs.com/gif.latex?X4,X5,X6"  />中的任何一个<img src="https://latex.codecogs.com/gif.latex?X" title="X" />，分类器将它分类到0的概率大于0.5。用数学表示就是：

<img src="https://latex.codecogs.com/gif.latex?P(Y=1|X,\theta)>0.5,&space;\&space;\&space;\&space;\&space;\&space;\&space;(X=X1,X2,X3)"  />

<img src="https://latex.codecogs.com/gif.latex?P(Y=0|X,\theta)>0.5,&space;\&space;\&space;\&space;\&space;\&space;\&space;(X=X4,X5,X6)"  />

以上式子中Y是类别，<img src="https://latex.codecogs.com/gif.latex?\theta" title="\theta" />为模型的参数,它可能是标量，可能是向量，取决于模型是什么。那模型究竟是什么呢？我们可以合理创造它——

我们看到这里求的是概率，所以模型应该是取值为<img src="https://latex.codecogs.com/gif.latex?(0,1)" />的函数，而我们正好有一个著名的函数Sigmoid,其取值正是为<img src="https://latex.codecogs.com/gif.latex?(0,1)" />：

![](/uploads/logistic_regression_pic_2.png)

函数表达式： <img src="https://latex.codecogs.com/gif.latex?sigmoid(z)=\frac{1}{1&plus;e^{-z}}" title="sigmoid(z)=\frac{1}{1+e^{-z}}" />

所以我们想到将线性特征嵌入Sigmoid中，设：

<img src="https://latex.codecogs.com/gif.latex?P(Y=1|X,\theta)=\frac{1}{1&plus;e^{-(\theta&space;_{0}x_{1}&plus;\theta&space;_{0}x_{2}&plus;...&plus;\theta&space;_{0}x_{n})}}=\frac{1}{1&plus;e^{-\theta&space;^{T}X}},&space;\&space;\&space;\&space;\&space;(X=X1,X2,X3)"  />&emsp;&emsp;&emsp;&emsp;①


BTW,此模型下，参数<img src="https://latex.codecogs.com/gif.latex?\theta=(\theta_{1},\theta_{2},...,\theta_{n})" title="\theta=(\theta_{1},\theta_{2},...,\theta_{n})" />

这个时候，可能有人问，为什么一定是<img src="https://latex.codecogs.com/gif.latex?\theta_{1}x_{1}&plus;\theta_{1}x_{2}&plus;...&plus;\theta_{1}x_{n}"  />这样的线性组合？当然不一定，你可以使用更复杂的组合。然而这里的线性组合是简单的且被证明work的，为什么不用呢？

回到我们的推导。Y=0的概率与Y=1的概率互补，既然Y=1的概率模型已经出来了，那么轻松得到：

<img src="https://latex.codecogs.com/gif.latex?P(Y=0|X,\theta)=1-P(Y=1|X,\theta)=1-\frac{1}{1&plus;e^{-\theta^{T}X}},&space;\&space;\&space;\&space;(X=X4,X5,X6)"  />&emsp;&emsp;&emsp;&emsp;②

这里我们可以do a trick，将①和②合并（之后会用到）：

<img src="https://latex.codecogs.com/gif.latex?P(Y|X,\theta)=(\frac{1}{1&plus;e^{-\theta^{T}X}})^{Y}(1-\frac{1}{1&plus;e^{-\theta^{T}X}})^{1-Y}" />&emsp;&emsp;&emsp;&emsp;③

### 损失函数

现在我们已经得到了概率模型。此时我们可以通过<font size="5">极大似然</font>的思维方式得到损失函数，即既然已经分类出来了，那么就不满足概率仅仅大于0.5，而是让分到这个类的概率尽可能大，如果我们的训练样本数为m个，那么我们希望m个概率的乘积尽可能大。所以似然函数是<img src="https://latex.codecogs.com/gif.latex?\prod_{i=1}^{m}P(y^{(i)}|x^{(i)},\theta)" />，我们需要最大化它。取负号就是最小化它，于是它损失函数就是损失函数：

<img src="https://latex.codecogs.com/gif.latex?L(\theta)=-\prod_{i=1}^{m}P(y^{(i)}|x^{(i)},\theta)" />


<img src="https://latex.codecogs.com/gif.latex?=-\prod_{i=1}^{m}(\frac{1}{1&plus;e^{-\theta^{T}x^{(i)}}})^{y^{(i)}}(1-\frac{1}{1&plus;e^{-\theta^{T}x^{(i)}}})^{1-y^{(i)}}" />

<img src="https://latex.codecogs.com/gif.latex?l(\theta)=ln(L(\theta))=-\sum_{i=1}^{m}ln[(\frac{1}{1&plus;e^{-\theta^{T}x^{(i)}}})^{y^{(i)}}(1-\frac{1}{1&plus;e^{-\theta^{T}x^{(i)}}})^{1-y^{(i)}}]"  />

<img src="https://latex.codecogs.com/gif.latex?=\sum_{i=1}^{m}[-y^{(i)}ln(\frac{1}{1&plus;e^{-\theta&space;^{T}x^{(i)}}})-(1-y^{(i)})ln(1-\frac{1}{1&plus;e^{-\theta&space;^{T}x^{(i)}}})]" />

### 损失函数求导

为了方便求导，设:

<img src="https://latex.codecogs.com/gif.latex?z=\theta^{T}x^{(i)}"/>

<img src="https://latex.codecogs.com/gif.latex?h(z)=\frac{1}{1&plus;e^{-z}}"  />

上两式求分别对<img src="https://latex.codecogs.com/gif.latex?\theta_{j}" />求导：
<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;z}{\partial\theta_{j}}=\frac{\partial&space;(\theta^{T}x^{(i)})}{\partial\theta_{j}}=\frac{\partial&space;(\theta_{1}x^{(i)}_{1}&plus;\theta_{1}x^{(i)}_{2}&plus;...&plus;\theta_{1}x^{(i)}_{n})}{\partial\theta_{j}}=x^{(i)}_{j}"  />

<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;h(z)}{\partial&space;z}=[(1&plus;e^{-z})^{-1}]'=-\frac{(e^{-z})'}{(1&plus;e^{-z})^2}=\frac{e^{-z}}{(1&plus;e^{-z})^{2}}"  />

<img src="https://latex.codecogs.com/gif.latex?=\frac{1&plus;e^{-z}-1}{(1&plus;e^{-z})^{2}}=\frac{1&plus;e^{-z}}{(1&plus;e^{-z})^2}-\frac{1}{(1&plus;e^{-z})^2}=\frac{1}{1&plus;e^{-z}}-\frac{1}{(1&plus;e^{-z})^2}" />

<img src="https://latex.codecogs.com/gif.latex?=h(z)[1-h(z)]" />

注：这里对<img src="https://latex.codecogs.com/gif.latex?\theta_{j}"  />是因为<img src="https://latex.codecogs.com/gif.latex?\theta" />是一个向量，我们最终是要求出该向量的所有维度的值。

将<img src="https://latex.codecogs.com/gif.latex?h(z)=\frac{1}{1&plus;e^{-z}}"  />代入损失函数:

<img src="https://latex.codecogs.com/gif.latex?l(\theta)=-\sum_{i=1}^{m}[y^{(i)}ln(h(z))&plus;(1-y^{(i)})ln(1-h(z))]" />


好了，准备就绪，开始对损失函数求导：
<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;l(\theta)}{\partial&space;\theta_{j}}=\frac{\partial&space;l(\theta)}{\partial&space;h}\frac{\partial&space;h}{\partial&space;z}\frac{\partial&space;z}{\partial&space;\theta_{j}}"  />

<img src="https://latex.codecogs.com/gif.latex?=-\sum_{i=1}^{m}[y^{(i)}\frac{1}{h(z)}-(1-y^{(i)})\frac{1}{1-h(z)}]\times&space;h(z)[1-h(z)]\times&space;x^{(i)}_{j}"  />

<img src="https://latex.codecogs.com/gif.latex?=-\sum_{i=1}^{m}\{y^{(i)}[1-h(z)]-(1-y^{(i)})h(z)\}\times&space;x^{(i)}_{j}" />

<img src="https://latex.codecogs.com/gif.latex?=-\sum_{i=1}^{m}[y^{(i)}-h(z)]\times&space;x^{(i)}_{j}" />

<img src="https://latex.codecogs.com/gif.latex?=-\sum_{i=1}^{m}[y^{(i)}-h(\theta^{T}x^{(i)})]\times&space;x^{(i)}_{j}"  />

至此，我们得到参数更新的迭代公式为：

<img src="https://latex.codecogs.com/gif.latex?\theta_{j}:=\theta_{j}-\alpha&space;\sum_{i=1}^{m}[y^{(i)}-h(\theta^{T}x^{(i)})]\times&space;x^{(i)}_{j}" />

接下来怎么办？——更新参数，训练分类器。训练过程由软件完成，这里就不手动推导了。




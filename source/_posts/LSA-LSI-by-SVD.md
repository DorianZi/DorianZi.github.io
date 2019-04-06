---
title: LSA/LSI by SVD 基于奇异值分解的潜在语义分析
date: 2019-03-25 08:09:26
tags: ["NLP","SVD","Algorithm"]
categories: Technic
---

## 问题的提出
在一个图书馆系统中，如果用户键入“计算机”，系统可能会返回诸如《计算机技术与应用》，《互联网通信技术》《C++编程理论》，《算法之美》，《电脑维修方法》等书籍。我们的知道“计算机” 与“互联网”，“编程”，“算法”，“电脑”有着类似的潜在语义，所以它们被筛选出来是可以理解的，但是系统是怎么知道它们有着类似的潜在语义呢？ 答案是LSA 或者 LSI

## LSA/LSI
LSA，即Latent Semantic Analysis，也有说是LSI，即Latent Semantic Indexing。它们表示的是同一个技术，以下用LSA来表示。

LSA是基于这个前提训练出来的：两个词如果在同一文本（一个书名为一个文本）中出现的次数一样，则认为这两个词含有同一语义，当然这是基于大量文档决定的。比如，系统“见过”《计算机编程方法》，《计算机云应用编程技术》，《编程原理及其计算机系统》，《计算机原理即编程实践》之后，能够学习到“计算机”和“编程”是同一语义的，于是当你要求它给出“计算机”相关书籍时，它能够返回《C++编程理论》，尽管该书名并没有“计算机”三个字。当然它也可能学习到“计算机”和“原理”是同一语义，这并不是我们想要的——要尽量避免这种情况，我们需要大量文档。下面我们就来讲解该学习方法：

我们考虑下面几本书的标题：
c1: Human machine interface for Lab ABC computer applications
c2: A survey of user opinion of computer system response time
c3: The EPS user interface management system
c4: System and human system engineering testing of EPS
c5: Relation of user-perceived response time to error measurement
m1: The generation of random, binary, unordered trees
m2: The intersection graph of paths in trees
m3: Graph minors IV: Widths of trees and well-quasi-ordering

首先构建词汇-文本矩阵：

![](/uploads/LSA_words_context_matrix.png)

将矩阵做SVD分解：

<img src="https://latex.codecogs.com/gif.latex?\underset{m\times&space;n}{A}&space;=\underset{m\times&space;m}{U}\underset{m\times&space;n}&space;{\sum}&space;\underset{n\times&space;n}{V^{T}}" title="\underset{m\times n}{A} =\underset{m\times m}{U}\underset{m\times n} {\sum} \underset{n\times n}{V^{T}}" />

<img src="https://latex.codecogs.com/gif.latex?U=\begin{bmatrix}&space;-0.22&space;&&space;-0.31&space;&&space;0.&space;&&space;0.41&space;&&space;-0.32&space;&&space;0.58&space;&&space;0.&space;&&space;0.&space;&&space;0.19&space;&&space;0.31&space;&&space;-0.25&space;&&space;-0.21&space;\\&space;-0.2&space;&&space;-0.15&space;&&space;0.&space;&&space;0.56&space;&&space;0.56&space;&&space;-0.09&space;&&space;0.&space;&&space;-0.&space;&&space;0.37&space;&&space;-0.22&space;&&space;0.08&space;&&space;0.34&space;\\&space;-0.24&space;&&space;0.17&space;&&space;0.&space;&&space;0.59&space;&&space;-0.29&space;&&space;-0.32&space;&&space;0.&space;&&space;-0.&space;&&space;-0.56&space;&&space;-0.09&space;&&space;0.17&space;&&space;-0.13&space;\\&space;-0.4&space;&&space;0.35&space;&&space;0.&space;&&space;-0.09&space;&&space;0.5&space;&&space;0.&space;&&space;0.&space;&&space;0.&space;&&space;-0.25&space;&&space;0.44&space;&&space;-0.43&space;&&space;-0.11&space;\\&space;-0.65&space;&&space;-0.39&space;&&space;0.&space;&&space;-0.34&space;&&space;-0.26&space;&&space;-0.18&space;&&space;0.&space;&&space;-0.&space;&&space;-0.08&space;&&space;-0.1&space;&&space;-0.11&space;&&space;0.43&space;\\&space;-0.26&space;&&space;0.45&space;&&space;0.&space;&&space;-0.07&space;&&space;-0.09&space;&&space;0.32&space;&&space;0.&space;&&space;0.&space;&&space;0.13&space;&&space;-0.72&space;&&space;-0.23&space;&&space;-0.17&space;\\&space;-0.26&space;&&space;0.45&space;&&space;0.&space;&&space;-0.07&space;&&space;-0.09&space;&&space;0.32&space;&&space;0.&space;&&space;-0.&space;&&space;0.13&space;&&space;0.28&space;&&space;0.66&space;&&space;0.28&space;\\&space;-0.31&space;&&space;-0.36&space;&&space;0.&space;&&space;-0.19&space;&&space;0.3&space;&&space;0.03&space;&&space;0.&space;&&space;-0.&space;&&space;-0.04&space;&&space;-0.12&space;&&space;0.46&space;&&space;-0.66&space;\\&space;-0.18&space;&&space;0.22&space;&&space;0.&space;&&space;0.02&space;&&space;-0.26&space;&&space;-0.56&space;&&space;0.&space;&&space;0.&space;&&space;0.64&space;&&space;0.19&space;&&space;-0.06&space;&&space;-0.3&space;\\&space;0.&space;&&space;0.&space;&&space;-0.74&space;&&space;0.&space;&&space;0.&space;&&space;0.&space;&&space;-0.59&space;&&space;-0.33&space;&&space;0.&space;&&space;0.&space;&&space;0.&space;&&space;0.&space;\\&space;0.&space;&&space;0.&space;&&space;-0.59&space;&&space;0.&space;&&space;0.&space;&&space;0.&space;&&space;0.33&space;&&space;0.74&space;&&space;0.&space;&&space;0.&space;&&space;0.&space;&&space;-0.&space;\\&space;0.&space;&&space;0.&space;&&space;-0.33&space;&&space;0.&space;&&space;0.&space;&&space;0.&space;&&space;0.74&space;&&space;-0.59&space;&&space;0.&space;&&space;0.&space;&&space;-0.&space;&&space;0.&space;\end{bmatrix}" />

<img src="https://latex.codecogs.com/gif.latex?\sum&space;=&space;\begin{bmatrix}&space;3.33&space;&&space;&&space;&&space;...&space;&&space;&&space;&&space;&&space;0\\&space;&&space;2.36&space;&&space;&&space;&&space;&&space;&&space;&\\&space;&&space;&&space;2.25&space;&&space;&&space;&&space;&&space;&\\&space;&&space;&&space;&&space;1.64&space;&&space;&&space;&&space;&\\&space;&&space;&&space;&&space;&&space;1.36&space;&&space;&&space;&\\&space;&&space;&&space;&&space;&&space;&&space;0.86&space;&&space;&\\&space;&&space;&&space;&&space;&&space;&&space;&&space;0.8&space;&&space;\\&space;&&space;&&space;&&space;&&space;&&space;&&space;&&space;0.55\\&space;0&space;&&space;&&space;&&space;...&space;&&space;&&space;&&space;&&space;0&space;\\&space;0&space;&&space;&&space;&&space;...&space;&&space;&&space;&&space;&&space;0\\&space;0&space;&&space;&&space;&&space;...&space;&&space;&&space;&&space;&&space;0\\&space;0&space;&&space;&&space;&&space;...&space;&&space;&&space;&&space;&&space;0&space;\end{bmatrix}"  />

<img src="https://latex.codecogs.com/gif.latex?V^{T}=&space;\begin{bmatrix}&space;-0.2&space;&&space;-0.6&space;&&space;-0.47&space;&&space;-0.55&space;&&space;-0.28&space;&&space;0.&space;&&space;0.&space;&&space;0.&space;\\&space;-0.12&space;&&space;0.53&space;&&space;-0.23&space;&&space;-0.61&space;&&space;0.53&space;&&space;0.&space;&&space;0.&space;&&space;0.&space;\\&space;0.&space;&&space;0.&space;&&space;0.&space;&&space;0.&space;&&space;0.&space;&&space;-0.33&space;&&space;-0.59&space;&&space;-0.74&space;\\&space;0.95&space;&&space;0.03&space;&&space;-0.04&space;&&space;-0.27&space;&&space;-0.14&space;&&space;0.&space;&&space;0.&space;&&space;0.&space;\\&space;-0.04&space;&&space;-0.36&space;&&space;0.81&space;&&space;-0.4&space;&&space;0.24&space;&&space;0.&space;&&space;0.&space;&&space;0.&space;\\&space;0.2&space;&&space;-0.48&space;&&space;-0.28&space;&&space;0.3&space;&&space;0.75&space;&&space;-0.&space;&&space;-0.&space;&&space;-0.&space;\\&space;-0.&space;&&space;-0.&space;&&space;-0.&space;&&space;-0.&space;&&space;-0.&space;&&space;-0.74&space;&&space;-0.33&space;&&space;0.59&space;\\&space;0.&space;&&space;0.&space;&&space;0.&space;&&space;0.&space;&&space;0.&space;&&space;-0.59&space;&&space;0.74&space;&&space;-0.33&space;\end{bmatrix}"  />


取3个奇异值进行降维，即k=3:

<img src="https://latex.codecogs.com/gif.latex?\underset{m\times&space;n}{A}&space;=\underset{m\times&space;k}{U}\underset{k\times&space;k}&space;{\sum}&space;\underset{k\times&space;n}{V^{T}}"  />

<img src="https://latex.codecogs.com/gif.latex?=&space;\begin{bmatrix}&space;-0.22&space;&&space;-0.31&space;&&space;0.&space;\\&space;-0.2&space;&&space;-0.15&space;&&space;0.&space;\\&space;-0.24&space;&&space;0.17&space;&&space;0.&space;\\&space;-0.4&space;&&space;0.35&space;&&space;0.\\&space;-0.65&space;&&space;-0.39&space;&&space;0.\\&space;-0.26&space;&&space;0.45&space;&&space;0.&space;\\&space;-0.26&space;&&space;0.45&space;&&space;0.\\&space;-0.31&space;&&space;-0.36&space;&&space;0.&space;\\&space;-0.18&space;&&space;0.22&space;&&space;0.&space;\\&space;0.&space;&&space;0.&&space;-0.74&space;\\&space;0.&space;&&space;0.&space;&&space;-0.59\\&space;0.&space;&&space;0.&space;&&space;-0.33&space;\end{bmatrix}&space;\begin{bmatrix}&space;3.33&space;&&space;0&space;&&space;0\\&space;0&space;&&space;2.36&space;&&space;0&space;\\&space;0&space;&&space;0&space;&&space;2.25&space;\end{bmatrix}&space;\begin{bmatrix}&space;-0.2&space;&&space;-0.6&space;&&space;-0.47&space;&&space;-0.55&space;&&space;-0.28&space;&&space;0.&space;&&space;0.&space;&&space;0.&space;\\&space;-0.12&space;&&space;0.53&space;&&space;-0.23&space;&&space;-0.61&space;&&space;0.53&space;&&space;0.&space;&&space;0.&space;&&space;0.&space;\\&space;0.&space;&&space;0.&space;&&space;0.&space;&&space;0.&space;&&space;0.&space;&&space;-0.33&space;&&space;-0.59&space;&&space;-0.74&space;\end{bmatrix}" />

好，我们完成了降维，数据变得简单了。看起来非常顺利，但是我们将词汇-文本矩阵用SVD分解后得到三个矩阵是什么意义呢， 降维操作降的是什么东西呢？

在[推荐系统的应用中](https://dorianzi.github.io/2019/03/20/recommender-system-by-SVD)，我们理解：<img src="https://latex.codecogs.com/gif.latex?U" />的行向量是各个user特征的向量，<img src="https://latex.codecogs.com/gif.latex?\sum" />的对角线值是权重，<img src="https://latex.codecogs.com/gif.latex?V^{T}"  />的列向量是各个item特征的向量，因为前几个权重占比非常大，所以仅保留user的前几个特征，和item的前几个特征以实现降维是合理的。

在现在讨论的潜在语义分析中，如何理解呢? 我们试着探索一下:

## 语义空间及其模型的引入

首先我们确定的是原始数据是词汇-文本矩阵，<img src="https://latex.codecogs.com/gif.latex?U" />是词汇-x矩阵，<img src="https://latex.codecogs.com/gif.latex?\sum" />是x-y矩阵，<img src="https://latex.codecogs.com/gif.latex?V^{T}"  />是y-文本矩阵：

![](/uploads/LSA_model_guess.png)

那么x和y分别是什么呢？我们<font size="5">人为地</font>认为，x="语义"，y="主题"，它的合理性在于：一个词汇可以对应不同的语义（<img src="https://latex.codecogs.com/gif.latex?U" />），而一个语义可以对应不同主题（<img src="https://latex.codecogs.com/gif.latex?\sum" />），而一个主题可以对应不同的文本（<img src="https://latex.codecogs.com/gif.latex?V^{T}"  />）。且它是可降维的，因为词汇对应的相关性低的语义可以忽略（<img src="https://latex.codecogs.com/gif.latex?U" />的降维），被忽略的语义对应的主题也被忽略（<img src="https://latex.codecogs.com/gif.latex?\sum" />的降维），文本对应的相关性低的主题可以忽略（<img src="https://latex.codecogs.com/gif.latex?V^{T}"  /> 的降维）

![](/uploads/LSA_model.png)


## 检索
在上面的计算中，我们已经得到了降维到3维的语义空间的模型，现在利用该模型计算检索需求。

求：当检索“human computer interaction”时，系统应该返回什么书籍？ 

计算如下：

检索项目在词汇-文本里面是列向量<img src="https://latex.codecogs.com/gif.latex?S=\begin{pmatrix}&space;1\\&space;0\\&space;1\\&space;0\\&space;0\\&space;0\\&space;0\\&space;0\\&space;0\\&space;0\\&space;0\\&space;0&space;\end{pmatrix}" /> , 将它投影到主题-文本空间的列向量(使用[公式](https://dorianzi.github.io/2019/03/20/recommender-system-by-SVD/#%E9%99%84%E5%BD%95))：

<img src="https://latex.codecogs.com/gif.latex?S'^{T}=S^{T}U(\sum&space;)^{-1}">

<img src="https://latex.codecogs.com/gif.latex?=&space;(1,0,1,0,0,0,0,0,0,0,0,0)&space;\begin{bmatrix}&space;-0.22&space;&&space;-0.31&space;&&space;0.&space;\\&space;-0.2&space;&&space;-0.15&space;&&space;0.&space;\\&space;-0.24&space;&&space;0.17&space;&&space;0.&space;\\&space;-0.4&space;&&space;0.35&space;&&space;0.\\&space;-0.65&space;&&space;-0.39&space;&&space;0.\\&space;-0.26&space;&&space;0.45&space;&&space;0.&space;\\&space;-0.26&space;&&space;0.45&space;&&space;0.\\&space;-0.31&space;&&space;-0.36&space;&&space;0.&space;\\&space;-0.18&space;&&space;0.22&space;&&space;0.&space;\\&space;0.&space;&&space;0.&&space;-0.74&space;\\&space;0.&space;&&space;0.&space;&&space;-0.59\\&space;0.&space;&&space;0.&space;&&space;-0.33&space;\end{bmatrix}&space;(&space;\begin{bmatrix}&space;3.33&space;&&space;0&space;&&space;0\\&space;0&space;&&space;2.36&space;&&space;0&space;\\&space;0&space;&&space;0&space;&&space;2.25&space;\end{bmatrix}&space;)^{-1}" />

<img src="https://latex.codecogs.com/gif.latex?=(-0.14,&space;-0.06,&space;0.)" />

=>

<img src="https://latex.codecogs.com/gif.latex?S'=\begin{pmatrix}&space;-0.14\\&space;-0.06\\&space;0.&space;\end{pmatrix}"  />

至此，我们得到了主题-文本空间的检索向量。 接下来计算该向量与此空间其它的向量（即<img src="https://latex.codecogs.com/gif.latex?V^{T}" />的列向量）的相似度，找出相似度最高的那个。

我们通过余弦相似度进行计算。余弦相似度定义为两个向量A,B的夹角：<img src="https://gss0.bdstatic.com/94o3dSag_xI4khGkpoWK1HF6hhy/baike/s%3D394/sign=20b5db49b7a1cd1101b674298d13c8b0/ac4bd11373f0820282c6ae4646fbfbedab641b76.jpg">

计算过程：

<img src="https://latex.codecogs.com/gif.latex?\underset{S'->v_{1}}{similarity}=&space;\frac{\begin{bmatrix}&space;-0.2\\&space;-0.12\\&space;0.&space;\end{bmatrix}\cdot&space;\begin{bmatrix}&space;-0.14\\&space;-0.06\\&space;0.&space;\end{bmatrix}&space;}{||\begin{bmatrix}&space;-0.2\\&space;-0.12\\&space;0.&space;\end{bmatrix}||\times&space;||\begin{bmatrix}&space;-0.14\\&space;-0.06\\&space;0.&space;\end{bmatrix}||}=0.991"  />

<img src="https://latex.codecogs.com/gif.latex?\underset{S'->v_{2}}{similarity}=0.428"  /> &emsp;&emsp; <img src="https://latex.codecogs.com/gif.latex?\underset{S'->v_{3}}{similarity}=0.998"  />&emsp;&emsp;<img src="https://latex.codecogs.com/gif.latex?\underset{S'->v_{4}}{similarity}=0.908"  />&emsp;&emsp;<img src="https://latex.codecogs.com/gif.latex?\underset{S'->v_{5}}{similarity}=0.081"  />&emsp;&emsp;<img src="https://latex.codecogs.com/gif.latex?\underset{S'->v_{6}}{similarity}=0.0"  />&emsp;&emsp;

<img src="https://latex.codecogs.com/gif.latex?\underset{S'->v_{7}}{similarity}=0.0"  />&emsp;&emsp;<img src="https://latex.codecogs.com/gif.latex?\underset{S'->v_{8}}{similarity}=0.0"  />

=> 

与<img src="https://latex.codecogs.com/gif.latex?S'"  />夹角（相似度）由小到大的向量排序：<img src="https://latex.codecogs.com/gif.latex?v_{3},v_{1},v_{4},v_{2},v_{5},v_{6},v_{7},v_{8}"  />，所以系统第一个将返回<img src="https://latex.codecogs.com/gif.latex?v_{3}"  />对应的书名“The EPS user interface management system”（与直观相违背？没关系，这是因为我们这里用的数据量太小，奇异值取个数也少，所以准确度有待商榷）。在实际中，系统可能会返回按照相关官渡排序的多个结果，且考虑到第5个结果开始相关系急剧下降，则就会返回前4个结果：

c3: The EPS user interface management system

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;↓

c1: Human machine interface for Lab ABC computer applications

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;↓

c4: System and human system engineering testing of EPS

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;↓

c2: A survey of user opinion of computer system response time


以上。

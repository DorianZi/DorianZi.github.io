---
title: Recommender System by SVD
date: 2019-03-20 17:13:32
tags:
---

## 引入

在之前的[SVD Decomposition](https://dorianzi.github.io/2019/03/09/matrix_SVD_decomposition/)一文中，我们介绍过，一个mxn的矩阵A可以分解为:

<img src="https://latex.codecogs.com/gif.latex?\underset{m\times&space;n}{A}=\underset{m\times&space;m}{U}\underset{m\times&space;n}{\sum}\underset{n\times&space;n}{V^{T}}">

其中<img src="https://latex.codecogs.com/gif.latex?U" title="U" />为<img src="https://latex.codecogs.com/gif.latex?AA^{T}">的特征矩阵，<img src="https://latex.codecogs.com/gif.latex?V" title="V" />为<img src="https://latex.codecogs.com/gif.latex?A^{T}A">的特征矩阵。同时我们通过选取部分奇异值及其对应的特征向量的方式实现对<img src="https://latex.codecogs.com/gif.latex?U,&space;\sum,V" title="U, \sum,V" />的降维：

<img src="https://github.com/DorianZi/algorithm_explained/blob/master/res/svd_cut_2.png?raw=true">

这一降维方式可以运用在[图像压缩](https://dorianzi.github.io/2019/03/10/image_compression_with_SVD/)中。

此文中，我们将继续探索SVD降维方法在<font size="5">推荐系统</font>中的应用。

## 推荐系统实践

### 全维推荐
我们考虑这样一个评分样本，6个user对5个item分别进行了评分，没有评分的计作0分：

<img src="https://github.com/DorianZi/algorithm_explained/raw/master/res/svd_recommender_data_2.png">

接下来，我们的目标是向user_5推荐item。应该向Ta推荐item_1，还是item_5呢? ——

这里采用的策略是：找到与user_5喜好最接近的用户，然后把该用户对item_1, item_5中评分最高的那一个推荐给user_5。于是问题转化为：<font size="4">哪个用户与user_5的喜好最接近？</font>

把用户所在行抽离出来，就会得到用户们的喜好向量：

<img src="https://latex.codecogs.com/gif.latex?user\_1^{T}&space;=&space;(1,5,0,5,4)"/>

<img src="https://latex.codecogs.com/gif.latex?user\_2^{T}&space;=&space;(5,4,4,3,2)"/>

<img src="https://latex.codecogs.com/gif.latex?user\_3^{T}&space;=&space;(0,4,0,0,5)"/>

<img src="https://latex.codecogs.com/gif.latex?user\_4^{T}&space;=&space;(4,4,1,4,0)"/>

<img src="https://latex.codecogs.com/gif.latex?user\_5^{T}&space;=&space;(0,4,3,5,0)"/>

<img src="https://latex.codecogs.com/gif.latex?user\_6^{T}&space;=&space;(2,4,3,5,3)"/>

然后我们要找出与<img src="https://latex.codecogs.com/gif.latex?user\_5^{T}" title="user\_5^{T}" />向量相似度最高的向量。

<font size="4">向量的相似度</font>是怎么评估的？—— 评估标准有很多种，常见的有：夹角余弦法，欧氏距离法等。这里我们使用欧式距离，即两个向量<img src="https://latex.codecogs.com/gif.latex?(x_{11},x_{12},...,x_{1n})" title="(x_{11},x_{12},...,x_{1n})" />和<img src="https://latex.codecogs.com/gif.latex?(x_{21},x_{22},...,x_{2n})" title="(x_{21},x_{22},...,x_{2n})" />之间的欧式距离为：

<img src="https://latex.codecogs.com/gif.latex?d=\sqrt{\sum_{i=1}^{n}(x_{1i}-x_{2i})^2}" title="d=\sqrt{\sum_{i=1}^{n}(x_{1i}-x_{2i})^2}" />

欧式距离越小，向量相似度越高。

通过计算得知，<img src="https://latex.codecogs.com/gif.latex?user\_6^{T}" title="user\_6^{T}" />与<img src="https://latex.codecogs.com/gif.latex?user\_5^{T}" title="user\_5^{T}" />相似度最高，所以user_6和user_5的喜好最接近，于是在item_1和item_5之间，将user_6评分更高的item_5推荐给user_5


### SVD降维推荐
以上，我们没有用到SVD。但是在实际应用中，该user-item数据表的维度远远大于6x5。在巨大维度下，逐一计算欧式距离成本太大。因此使用SVD进行降维，再计算欧式距离，是推荐系统里常用的方法。

(待更新...）




## 参考
http://etd.lib.metu.edu.tr/upload/12612129/index.pdf



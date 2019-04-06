---
title: Recommender System by SVD 基于SVD的推荐系统
date: 2019-03-20 17:13:32
categories: ["Technic"]
tags: ["Algorithm", "SVD"]
---

## 引入

在之前的[SVD Decomposition](https://dorianzi.github.io/2019/03/09/matrix_SVD_decomposition/)一文中，我们介绍过，一个mxn的矩阵A可以分解为:

<img src="https://latex.codecogs.com/gif.latex?\underset{m\times&space;n}{A}=\underset{m\times&space;m}{U}\underset{m\times&space;n}{\sum}\underset{n\times&space;n}{V^{T}}">

其中<img src="https://latex.codecogs.com/gif.latex?U" title="U" />为<img src="https://latex.codecogs.com/gif.latex?AA^{T}">的特征矩阵，<img src="https://latex.codecogs.com/gif.latex?V" title="V" />为<img src="https://latex.codecogs.com/gif.latex?A^{T}A">的特征矩阵。同时我们通过选取部分奇异值及其对应的特征向量的方式实现对<img src="https://latex.codecogs.com/gif.latex?U,&space;\sum,V" title="U, \sum,V" />的降维：

<img src="https://github.com/DorianZi/algorithm_explained/blob/master/res/svd_cut_2.png?raw=true">

这一降维方式可以运用在[图像压缩](https://dorianzi.github.io/2019/03/10/image_compression_with_SVD/)中。

此文中，我们将继续探索SVD降维方法在<font size="5">推荐系统</font>中的应用。

## 推荐系统实践

### 问题的提出
我们考虑这样一个评分样本，6个user对5个item分别进行了评分，没有评分的计作0分：

<img src="https://github.com/DorianZi/algorithm_explained/raw/master/res/svd_recommender_data_2.png">

接下来，我们的目标是向user_5推荐item。应该向Ta推荐item_1，还是item_5呢? ——

这里采用的策略是：找到与user_5喜好最接近的用户，然后把该用户对item_1, item_5中评分最高的那一个推荐给user_5。于是问题转化为：<font size="4">哪个用户与user_5的喜好最接近？</font>

在实际应用中，user-item数据表的维度远远大于6x5。在巨大维度下，计算成本太高，使用SVD进行降维后再计算，是推荐系统里常用的方法。

### SVD求解
首先将样本数据看做一个矩阵6x5的矩阵并进行SVD分解：

<img src="https://latex.codecogs.com/png.latex?\underset{\underset{m&space;\times&space;n}{A}}{\begin{bmatrix}&space;1&space;&&space;5&&space;0&&space;5&&space;4\\&space;5&space;&&space;4&&space;4&space;&3&space;&2&space;\\&space;0&&space;4&&space;0&&space;0&&space;5\\&space;4&&space;4&&space;1&&space;4&&space;0\\&space;0&&space;4&&space;3&&space;5&&space;0\\&space;2&&space;4&&space;3&&space;5&&space;3&space;\end{bmatrix}}&space;=&space;\underset{\underset{m&space;\times&space;m}{U}}{\begin{bmatrix}&space;-0.46&space;&&space;0.40&space;&&space;0.30&space;&&space;-0.43&space;&&space;0.32&space;&&space;-0.50\\&space;-0.46&space;&&space;-0.30&space;&&space;-0.65&space;&&space;0.28&space;&&space;0.02&space;&&space;-0.44&space;\\&space;-0.25&space;&&space;0.75&space;&&space;-0.28&space;&&space;0.16&space;&&space;-0.46&space;&&space;0.22\\&space;-0.38&space;&&space;-0.35&space;&&space;-0.13&space;&&space;-0.68&space;&&space;-0.32&space;&&space;0.38&space;\\&space;-0.38&space;&&space;-0.24&space;&&space;0.62&space;&&space;0.38&space;&&space;-0.50&space;&&space;-0.13\\&space;-0.48&space;&&space;-0.01&space;&&space;0.10&space;&&space;0.31&space;&&space;0.57&space;&&space;0.59&space;\end{bmatrix}&space;}&space;\underset{\underset{m&space;\times&space;n}{\sum}}{\begin{bmatrix}&space;16.47&space;&&space;0&space;&&space;0&space;&&space;0&space;&&space;0\\&space;0&space;&&space;6.21&space;&&space;0&space;&&space;0&space;&&space;0\\&space;0&&space;0&space;&&space;4.40&space;&&space;0&space;&&space;0&space;\\&space;0&&space;0&space;&&space;0&space;&&space;2.90&space;&&space;0&space;\\&space;0&&space;0&space;&&space;0&space;&&space;0&space;&&space;1.58&space;\\&space;0&&space;0&space;&&space;0&space;&&space;0&space;&&space;0&space;\end{bmatrix}&space;}&space;\underset{\underset{n&space;\times&space;n}{V^{T}}}{&space;\begin{bmatrix}&space;-0.32&space;&&space;-0.61&space;&&space;-0.29&space;&&space;-0.58&space;&&space;-0.33\\&space;-0.41&space;&&space;0.22&space;&&space;-0.38&space;&&space;-0.26&space;&&space;0.76\\&space;-0.74&space;&&space;0.03&space;&&space;-0.13&space;&&space;0.60&space;&&space;-0.27&space;\\&space;-0.39&space;&&space;-0.12&space;&&space;0.87&space;&&space;-0.20&space;&&space;0.19\\&space;0.17&space;&&space;-0.75&space;&&space;-0.03&space;&&space;0.45&space;&&space;0.45&space;\end{bmatrix}&space;}" title="\underset{\underset{m \times n}{A}}{\begin{bmatrix} 1 & 5& 0& 5& 4\\ 5 & 4& 4 &3 &2 \\ 0& 4& 0& 0& 5\\ 4& 4& 1& 4& 0\\ 0& 4& 3& 5& 0\\ 2& 4& 3& 5& 3 \end{bmatrix}} = \underset{\underset{m \times m}{U}}{\begin{bmatrix} -0.46 & 0.40 & 0.30 & -0.43 & 0.32 & -0.50\\ -0.46 & -0.30 & -0.65 & 0.28 & 0.02 & -0.44 \\ -0.25 & 0.75 & -0.28 & 0.16 & -0.46 & 0.22\\ -0.38 & -0.35 & -0.13 & -0.68 & -0.32 & 0.38 \\ -0.38 & -0.24 & 0.62 & 0.38 & -0.50 & -0.13\\ -0.48 & -0.01 & 0.10 & 0.31 & 0.57 & 0.59 \end{bmatrix} } \underset{\underset{m \times n}{\sum}}{\begin{bmatrix} 16.47 & 0 & 0 & 0 & 0\\ 0 & 6.21 & 0 & 0 & 0\\ 0& 0 & 4.40 & 0 & 0 \\ 0& 0 & 0 & 2.90 & 0 \\ 0& 0 & 0 & 0 & 1.58 \\ 0& 0 & 0 & 0 & 0 \end{bmatrix} } \underset{\underset{n \times n}{V^{T}}}{ \begin{bmatrix} -0.32 & -0.61 & -0.29 & -0.58 & -0.33\\ -0.41 & 0.22 & -0.38 & -0.26 & 0.76\\ -0.74 & 0.03 & -0.13 & 0.60 & -0.27 \\ -0.39 & -0.12 & 0.87 & -0.20 & 0.19\\ 0.17 & -0.75 & -0.03 & 0.45 & 0.45 \end{bmatrix} }" />

对于评分矩阵，该分解意味着什么呢？我们来计算一下user_3对item_2的评分

<img src="https://github.com/DorianZi/algorithm_explained/blob/master/res/recommender_rate_dot.png?raw=true">

=>

<img src="https://latex.codecogs.com/png.latex?4&space;=&space;(-0.25)\times&space;16.47\times&space;(-0.61)&plus;0.75\times&space;6.21\times&space;0.22&plus;(-0.28)\times&space;4.4\times&space;0.03&plus;0.16\times&space;2.9\times&space;(-0.12)&plus;(-0.46)\times&space;1.58\times&space;(-0.75)" title="4 = (-0.25)\times 16.47\times (-0.61)+0.75\times 6.21\times 0.22+(-0.28)\times 4.4\times 0.03+0.16\times 2.9\times (-0.12)+(-0.46)\times 1.58\times (-0.75)" />

也就是说A中各个位置的元素来自于<img src="https://latex.codecogs.com/png.latex?U" title="U" />的对应位置行向量和<img src="https://latex.codecogs.com/png.latex?V^{T}" title="V^{T}" />的对应位置列向量按照奇异值进行加权点积。

将<img src="https://latex.codecogs.com/png.latex?U" title="U" />的行向量看作user向量在新特征下的表示，<img src="https://latex.codecogs.com/png.latex?V^{T}" title="V^{T}" />的列向量看作item向量在新特征下的表示，同时只截取权重最大的前两个特征，对数据进行降维：

<img src="https://github.com/DorianZi/algorithm_explained/blob/master/res/recommender_system_svd_feature.png?raw=true">

=>

<img src="https://github.com/DorianZi/algorithm_explained/blob/master/res/recommender_system_svd_feature_cut.png?raw=true">

至此，实现了降维。

我们在此维度下，寻找与user_5最接近的user。每个用户向量抽离出来是2维向量：

<img src="https://latex.codecogs.com/gif.latex?user\_1^{T}&space;=&space;(-0.46,0.40)"/>

<img src="https://latex.codecogs.com/gif.latex?user\_2^{T}&space;=&space;(-0.46，-0.30)"/>

<img src="https://latex.codecogs.com/gif.latex?user\_3^{T}&space;=&space;(-0.25,0.75)"/>

<img src="https://latex.codecogs.com/gif.latex?user\_4^{T}&space;=&space;(-0.38,-0.35)"/>

<img src="https://latex.codecogs.com/gif.latex?user\_5^{T}&space;=&space;(-0.38，-0.24)"/>

<img src="https://latex.codecogs.com/gif.latex?user\_6^{T}&space;=&space;(-0.48,-0.01)"/>

然后我们要找出与<img src="https://latex.codecogs.com/gif.latex?user\_5^{T}" title="user\_5^{T}" />向量相似度最高的向量。

<font size="4">向量的相似度</font>是怎么评估的？—— 评估标准有很多种，常见的有：夹角余弦法，欧氏距离法等。这里我们使用欧式距离，即两个向量<img src="https://latex.codecogs.com/gif.latex?(x_{11},x_{12},...,x_{1n})" title="(x_{11},x_{12},...,x_{1n})" />和<img src="https://latex.codecogs.com/gif.latex?(x_{21},x_{22},...,x_{2n})" title="(x_{21},x_{22},...,x_{2n})" />之间的欧式距离为：

<img src="https://latex.codecogs.com/gif.latex?d=\sqrt{\sum_{i=1}^{n}(x_{1i}-x_{2i})^2}" title="d=\sqrt{\sum_{i=1}^{n}(x_{1i}-x_{2i})^2}" />

欧式距离越小，向量相似度越高。

通过计算得知，<img src="https://latex.codecogs.com/gif.latex?user\_2^{T}" title="user\_2^{T}" />与<img src="https://latex.codecogs.com/gif.latex?user\_5^{T}" title="user\_5^{T}" />相似度最高，所以user_2和user_5的喜好最接近。

回到原始矩阵看user_2对item_1和item_5的评分：item_1评分，更高。 因此将item_1推荐给user_5。

## 向新用户推荐item
上面求解的是向已知用户推荐item。如果现在有一个新的用户<img src="https://latex.codecogs.com/gif.latex?X=(2,0,5,2,0)" title="user\_new^{T}=(2,0,5,2,0)" />，它的数据并不在上面的样本矩阵里，怎么办？
容易想到的是，直接加入原样本数据进行扩展再求解，但是这样做在大数据情况下时间消耗太大。我们需要基于已经有的模型对新加入的用户进行item推荐。

首先，将新用户投影到降维后的user向量空间中去（为什么是这个算法？~~我也在找寻答案~~ 请见<font size="4">**附录**</font>）：

<img src="https://latex.codecogs.com/gif.latex?X_{k}^{T}&space;=&space;X^{T}\underset{n\times&space;k}{V}\underset{k\times&space;k}{\sum}^{-1}" />

<img src="https://latex.codecogs.com/gif.latex?X_{2}^{T}&space;=(2,0,5,2,0)\begin{bmatrix}&space;-0.32&space;&&space;-0.41\\&space;-0.61&&space;0.22&space;\\&space;-0.29&space;&&space;-0.38&space;\\&space;-0.58&space;&&space;-0.26&space;\\&space;-0.33&space;&&space;0.76&space;\end{bmatrix}&space;\begin{bmatrix}&space;16.47&space;&&space;0\\&space;0&space;&&space;6.21&space;\end{bmatrix}^{-1}=(-0.20,-0.52)" />

接下来通过欧式距离或者其它方式,找出<img src="https://latex.codecogs.com/gif.latex?user\_1^{T}" /> ~ <img src="https://latex.codecogs.com/gif.latex?user\_1^{T}" />中与<img src="https://latex.codecogs.com/gif.latex?X_{2}^{T}}" />相似度最该的且对item_2和item_5评过分的user, 然后把该user对item_2和item_5中评分更高的item推荐给<img src="https://latex.codecogs.com/gif.latex?X" title="X" />用户

以上。

## 附录
将SVD分解的原始矩阵A和分解矩阵U都写成行向量形式：

<img src="https://latex.codecogs.com/gif.latex?A=U\sum&space;V^{T}"  />

=>

<img src="https://latex.codecogs.com/gif.latex?\begin{pmatrix}&space;A_{1}\\&space;A_{2}\\&space;...\\&space;A_{m}\\&space;\end{pmatrix}=&space;\begin{pmatrix}&space;u_{1}\\&space;u_{2}\\&space;...\\&space;u_{m}\\&space;\end{pmatrix}&space;\sum&space;V^{T}&space;=&space;\begin{pmatrix}&space;u_{1}\sum&space;V^{T}\\&space;u_{2}\sum&space;V^{T}\\&space;...\\&space;u_{m}\sum&space;V^{T}\\&space;\end{pmatrix}"  />

=>

<img src="https://latex.codecogs.com/gif.latex?A_{i}&space;=&space;u_{i}\sum&space;V^{T}\&space;,\&space;\&space;\&space;\&space;i=1,2,...,m" />

=>

<img src="https://latex.codecogs.com/gif.latex?u_{i}=A_{i}(V^{T})^{-1}(\sum)&space;^{-1}" title="u_{i}=A_{i}(V^{T})^{-1}(\sum) ^{-1}" />

=>

<img src="https://latex.codecogs.com/gif.latex?u_{i}=A_{i}V(\sum)&space;^{-1}" title="u_{i}=A_{i}V(\sum) ^{-1}" />

上述即原始user向量向新特征下的user向量的转换公式。

将SVD分解的原始矩阵A和分解矩阵<img src="https://latex.codecogs.com/gif.latex?V^{T}" />都写成列向量形式：

<img src="https://latex.codecogs.com/gif.latex?A=U\sum&space;V^{T}"  />

=>

<img src="https://latex.codecogs.com/gif.latex?(A_{1},A_{2},...,A_{n})=U\sum&space;(v_{1},v_{2},...,v_{n})"  />

=>

<img src="https://latex.codecogs.com/gif.latex?A_{i}=U\sum&space;v_{i}\&space;,\&space;\&space;\&space;\&space;i=1,2,...,n"  />

=>

<img src="https://latex.codecogs.com/gif.latex?v_{i}=(\sum&space;)^{-1}U^{-1}A_{i}" />

=>

<img src="https://latex.codecogs.com/gif.latex?v^{T}_{i}=A^{T}_{i}(U^{-1})^{T}((\sum&space;)^{-1})^{T}" />

=>

<img src="https://latex.codecogs.com/gif.latex?v^{T}_{i}=A^{T}_{i}U(\sum&space;)^{-1}" />

上述即原始item向量向新特征下的item向量的转换公式。


## 参考
http://etd.lib.metu.edu.tr/upload/12612129/index.pdf


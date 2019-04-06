---
title: Relationship between PCA and SVD 主成分分析和奇异值分解的关系
date: 2019-03-23 15:40:19
categories: ["Technic"]
tags: ["Algorithm", "SVD", "PCA"]
---

我们有一个样本集：
![](/uploads/pca_vs_svd.png)
我们不妨把它当做一个m个用户对n个物品评分的矩阵。为了PCA计算方便，就把它当做一个已经去中心化的数据矩阵。

## SVD的矩阵乘转置<=>PCA的协方差矩阵
我们先作一小步SVD分解：
按照<font size="4">SVD分解</font>的方法，我们要获得<img src="https://latex.codecogs.com/png.latex?\underset{m\times&space;n}{A}=\underset{m\times&space;m}{U}\underset{m\times&space;n}{\sum}&space;\underset{n\times&space;n}{V^{T}}" />中的U，需要对<img src="https://latex.codecogs.com/png.latex?AA^{T}" />进行特征值分解。

先来看看<img src="https://latex.codecogs.com/png.latex?AA^{T}" />长什么样子：

<img src="https://latex.codecogs.com/png.latex?\underset{m\times&space;m}{AA^{T}}=(x_{1},x_{2},...,x_{m})\begin{pmatrix}&space;x_{1}^{T}\\&space;x_{2}^{T}\\&space;...\\&space;x_{m}^{T}&space;\end{pmatrix}=\sum_{i=1}^{m}x_{i}x_{i}^{T}" />

看着很眼熟？好，我们再来做一小步PCA：

按照<font size="4">PCA</font>的方法，假设我们以行向量（用户向量）为目标数据，item为维度，要对item数量进行降维，那么我们首先要计算已经去中心化的A的协方差矩阵：

<img src="https://latex.codecogs.com/png.latex?\underset{n\times&space;n}{C}=\begin{bmatrix}&space;Cov(item\_1,item\_1)&space;&&space;Cov(item\_1,item\_2)&&space;...&&space;Cov(item\_1,item\_n)\\&space;Cov(item\_2,item\_1)&space;&&space;Cov(item\_2,item\_2)&&space;...&&space;Cov(item\_2,item\_n)&space;\\&space;...&space;&&space;...&&space;...&&space;...\\&space;Cov(item\_n,item\_1)&space;&&space;Cov(item\_2,item\_2)&&space;...&&space;Cov(item\_n,item\_n)&space;\end{bmatrix}"  />

<img src="https://latex.codecogs.com/gif.latex?=\frac{1}{m-1}&space;\begin{bmatrix}&space;x_{11}x_{11}+x_{21}x_{21}+...+x_{m1}x_{m1}&&space;x_{11}x_{12}+x_{21}x_{22}+...+x_{m1}x_{m2}&&space;...&space;&&space;x_{11}x_{1n}+x_{21}x_{2n}+...+x_{m1}x_{mn}\\&space;x_{12}x_{11}+x_{22}x_{21}+...+x_{m2}x_{m1}&&space;x_{12}x_{12}+x_{22}x_{22}+...+x_{m2}x_{m2}&&space;...&space;&&space;x_{12}x_{1n}+x_{22}x_{2n}+...+x_{m2}x_{mn}\\&space;&&space;&...&space;&&space;\\&space;x_{1n}x_{11}+x_{2n}x_{21}+...+x_{mn}x_{m1}&&space;x_{1n}x_{12}+x_{2n}x_{22}+...+x_{mn}x_{m2}&&space;...&space;&&space;x_{1n}x_{1n}+x_{2n}x_{2n}+...+x_{mn}x_{mn}&space;\end{bmatrix}">

<img src="https://latex.codecogs.com/gif.latex?=\frac{1}{m-1}\sum_{i=1}^{n}x_{i}x_{i}^{T}">

去掉常数项，这不就是<img src="https://latex.codecogs.com/png.latex?AA^{T}" />嘛！

至此，得：<font size="4">SVD中的<img src="https://latex.codecogs.com/png.latex?AA^{T}" />就是PCA中以A的列为维度的协方差</font>

同理，得：<font size="4">SVD中的<img src="https://latex.codecogs.com/png.latex?A^{T}A" />就是PCA中以A的行为维度的协方差</font>

## SVD的奇异矩阵<=>PCA的特征向量
我们继续SVD，也就是求<img src="https://latex.codecogs.com/png.latex?AA^{T}" />的特征向量<img src="https://latex.codecogs.com/png.latex?(u_{1},u_{2},...,u_{n})"/>来表示原矩阵：

<img src="https://latex.codecogs.com/png.latex?\underset{m\times&space;n}{A}=\underset{m\times&space;m}{(u_{1},u_{2},...,u_{m})}\underset{m\times&space;n}{\sum}\underset{n\times&space;n}{V^{T}}" title="\underset{m\times n}{A}=\underset{m\times m}{(u_{1},u_{2},...,u_{m})}\underset{m\times n}{\sum}\underset{n\times n}{V^{T}}" />

同时我们也继续PCA,也就是求协方差矩阵的特征向量<img src="https://latex.codecogs.com/png.latex?(u_{1},u_{2},...,u_{n})"/>并它们此为新的主元，写出新主元下的矩阵(即<img src="https://latex.codecogs.com/png.latex?UA" />)：
![](/uploads/pca_vs_svd_after_pca.png)

当然, SVD中通过<img src="https://latex.codecogs.com/png.latex?A^{T}A" />求得的右奇异矩阵<img src="https://latex.codecogs.com/png.latex?V^{T}" title="V^{T}" />，与PCA中以行为维度的协方差求得特征向量也有上述的对应关系

所以在实际运算中，我们可以通过求SVD来求解PCA，也可以通过PCA来求解SVD。事实上很多运算库如scikit-learn的PCA算法实现就是用的SVD方法。


## SVD的双向降维<=>PCA的单向降维

基于上面的求解结果，我们来进行最后的降维操作。这里以左奇异矩阵为例，右奇异矩阵请自行补脑。

使用SVD降维到k维时，会把左奇异矩阵截取(同时也会截取奇异值矩阵和右奇异矩阵，但是这里我们只关心左奇异矩阵)到前k个列向量<img src="https://latex.codecogs.com/png.latex?(u_{1},u_{2},...,u_{k})" title="(u_{1},u_{2},...,u_{k})" />:

<img src="https://latex.codecogs.com/png.latex?\underset{m\times&space;n}{A}=\underset{m\times&space;k}{(u_{1},u_{2},...,u_{k})}\underset{k\times&space;k}{\sum}\underset{k\times&space;n}{V^{T}}"  />

使用PCA降维到k维时，会选择特征矩阵的前k个维度来表示原数据:
![](/uploads/pca_vs_svd_pca_cut.png)

由此，我们可以认为，SVD对左奇异矩阵的降维就是对user向量的降维，而对右奇异矩阵的降维，就是对item向量的降维。SVD分解在推荐系统里基于<font size="4">这个结论</font>才能顺利进行下去


## 总结
1) SVD可以做双向降维，它将行向量（user)和列向量(item)都进行降维了; PCA只能做单向降维，需要一开始指定维度是列(目标是user向量)还是行(目标是item向量)，再进行降维操作
2) SVD相当于在行向量和列向量分别进行了PCA，并保存新主元到左奇异矩阵和右奇异矩阵中




---
title: Perceptron Classifier 感知机分类器
date: 2019-03-29 02:27:05
tags: ["Algorithm"]
categories: Technic
---

## 感知机模型

在n维度空间，有m个数据组成的线性可分的数据集<img src="https://latex.codecogs.com/gif.latex?\begin{bmatrix}&space;x_{11}&space;&&space;x_{12}&space;&&space;...&space;&&space;x_{1n}&space;&|&space;\&space;\&space;C_{1}\\&space;x_{21}&space;&&space;x_{22}&space;&&space;...&space;&&space;x_{2n}&space;&|&space;\&space;\&space;C_2\\&space;x_{31}&space;&&space;x_{32}&space;&&space;...&space;&&space;x_{3n}&space;&|&space;\&space;\&space;C_3\\&space;x_{41}&space;&&space;x_{42}&space;&&space;...&space;&&space;x_{4n}&space;&|&space;\&space;\&space;C_4\\&space;...&space;&&space;...&space;&&space;...&space;&&space;...&space;&|&space;\&space;\&space;\&space;...\\&space;x_{m1}&space;&&space;x_{m2}&space;&&space;...&space;&&space;x_{mn}&space;&\&space;\&space;|&space;\&space;\&space;\&space;C_m&space;\end{bmatrix}"  />，每个数据点都有对应的类别<img src="https://latex.codecogs.com/gif.latex?C_i" />，但一共只有两个类别：<img src="https://latex.codecogs.com/gif.latex?C_1=1,C_2=1,C_3=-1,C_4=1,...,C_m=-1"  />,则存在一个超平面<img src="https://latex.codecogs.com/gif.latex?WX^{T}&plus;b=0"  />，( 其中<img src="https://latex.codecogs.com/gif.latex?W=(w_{1},w_{2},...,w_{n})"  /> )能够准确地将所有的C1和C2数据区分开来，使得他们分别在在超平面的两边。

感知机只针对线性可分的数据，超平面在二维空间是一条线，在三维空间就是一个平面。

线性可分：
![](/uploads/Perceptron_1.gif)

线性不可分：
![](/uploads/Perceptron_2.gif)

超平面分类的判别方法是通过带入计算正负，比如：

<img src="https://latex.codecogs.com/gif.latex?W\begin{pmatrix}&space;x_{11}\\&space;x_{12}\\&space;...\\&space;x_{1n}&space;\end{pmatrix}&plus;b>0" /> 作为1类

<img src="https://latex.codecogs.com/gif.latex?W\begin{pmatrix}&space;x_{31}\\&space;x_{32}\\&space;...\\&space;x_{3n}&space;\end{pmatrix}&plus;b<0" /> 作为-1类


## 感知机的损失函数

感知机的学习过程就是将分错类的数据不断纠正的过程。设分类器对<img src="https://latex.codecogs.com/gif.latex?X_i" />判定的类别是<img src="https://latex.codecogs.com/gif.latex?Y_i"  /> ,而它实际的类别是<img src="https://latex.codecogs.com/gif.latex?C_i" title="C_i" /> ，完美情况下当然希望所有点都能满足<img src="https://latex.codecogs.com/gif.latex?Y_i=C_i"  /> 。

即使不完美我们也知道一个好的分类器会有这两个特性：被分错的点尽可能少，即使被分错了，也距离超平面非常近。能够描述这两个的特性的，就是被分错的点到超平面的距离之和。所以可以把这个距离之和作为损失函数。

点到超平面的距离为：<img src="https://latex.codecogs.com/gif.latex?\frac{(WX^T_i&plus;b)}{\|W\|}"  />, 注意它是有正负的，1类的点到超平面的距离为正，-1类的点为负。我们关注被分错的点，不管是1类点被错分到-1类，还是-1类点被错分到1类，他们的距离取绝对值都是：<img src="https://latex.codecogs.com/gif.latex?-Y_i\frac{(WX^T_i&plus;b)}{\|W\|}" />

设所有分错的点的几何是<img src="https://latex.codecogs.com/gif.latex?M" />,那么所有分错的点到超平面的距离和是：

<img src="https://latex.codecogs.com/gif.latex?L(W,b)=-\sum&space;_{X_i\epsilon&space;M}Y_i\frac{(WX^T_i&plus;b)}{\|W\|}"  />

忽略常数项 =>

<img src="https://latex.codecogs.com/gif.latex?L(W,b)=-\sum&space;_{X_i\epsilon&space;M}Y_i(WX^T_i&plus;b)"  />

对损失函数求导：

<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;L(W,b)}{\partial&space;W}=-\sum_{X_i\epsilon&space;M}^{&space;}Y_iX_i^T" title="\frac{\partial L(W,b)}{\partial W}=-\sum_{X_i\epsilon M}^{ }Y_iX_i^T" />

<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;L(W,b)}{\partial&space;b}=-\sum_{X_i\epsilon&space;M}^{&space;}Y_i" title="\frac{\partial L(W,b)}{\partial b}=-\sum_{X_i\epsilon M}^{ }Y_i" />

也就是说参数更新公式为：

<img src="https://latex.codecogs.com/gif.latex?W:=W&plus;\eta&space;\sum_{X_i\epsilon&space;M}^{&space;}Y_iX_i^T" title="W:=W+\alpha Y_iX_i^T" />

<img src="https://latex.codecogs.com/gif.latex?b:=b&plus;\eta&space;\sum_{X_i\epsilon&space;M}^{&space;}Y_i" title="b:=b+\beta Y_i" />

<img src="https://latex.codecogs.com/gif.latex?W" title="W" />的更新如果细化到每个维度，就是：

<img src="https://latex.codecogs.com/gif.latex?w_j:=w_j&plus;\eta&space;\sum_{X_i\epsilon&space;M}^{&space;}Y_iX_ij" title="w_j:=w_j+\eta Y_iX_ij" />


## 学习过程

输入：m个n维度的线性可分的已经标好label的数据集<img src="https://latex.codecogs.com/gif.latex?\begin{bmatrix}&space;x_{11}&space;&&space;x_{12}&space;&&space;...&space;&&space;x_{1n}&space;&|&space;\&space;\&space;C_{1}\\&space;x_{21}&space;&&space;x_{22}&space;&&space;...&space;&&space;x_{2n}&space;&|&space;\&space;\&space;C_2\\&space;x_{31}&space;&&space;x_{32}&space;&&space;...&space;&&space;x_{3n}&space;&|&space;\&space;\&space;C_3\\&space;x_{41}&space;&&space;x_{42}&space;&&space;...&space;&&space;x_{4n}&space;&|&space;\&space;\&space;C_4\\&space;...&space;&&space;...&space;&&space;...&space;&&space;...&space;&|&space;\&space;\&space;\&space;...\\&space;x_{m1}&space;&&space;x_{m2}&space;&&space;...&space;&&space;x_{mn}&space;&\&space;\&space;|&space;\&space;\&space;\&space;C_m&space;\end{bmatrix}"  />， <img src="https://latex.codecogs.com/gif.latex?C_i" title="C_i" />只有1或-1两种label。

输出：<img src="https://latex.codecogs.com/gif.latex?W,b" />

1) 选择初始值 <img src="https://latex.codecogs.com/gif.latex?W_0,b_0"  />构建初始感知机分类器
2) 选择数据点<img src="https://latex.codecogs.com/gif.latex?(X_i,C_i)" />进行分类，判断是否为分错的点，即如果<img src="https://latex.codecogs.com/gif.latex?Y_i(WX^T_i&plus;b)<0" />则为分错的点。否则循环2)
3) 更新参数，因为每次只取一个点来更新参数，所以参数迭代公式可以简化为：

<img src="https://latex.codecogs.com/gif.latex?W:=W&plus;\eta&space;Y_iX_i^T" title="W:=W+\alpha Y_iX_i^T" />

<img src="https://latex.codecogs.com/gif.latex?b:=b&plus;\eta&space;Y_i" title="b:=b+\beta Y_i" />

4）回到2）直到找不到分错的点


以上。

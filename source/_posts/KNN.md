---
title: KNN K近邻分类器
date: 2019-03-29 10:04:12
tags: ["Algorithm"]
categories: Technic
---

## KNN

KNN即K-NearestNeighbor分类器，是最简单的最初级的分类器。其核心思想是找到与待分类点最接近的K个样本点，然后把K个样本点中指向最多的那个类别认为是待分类点的类别。图示如下：

![](/uploads/KNN_1.png)

<font color="Green">绿色</font>圆圈为待分类的数据，<font color="Blue">蓝色</font>方块为样本中的一类，<font color="Red">红色</font>三角为另一类。令K=4，也就是选取最近的4个样本数据，发现蓝色方块类别更多，则将待分类的数据分类到蓝色方块。

我们考虑“最近”一般采用欧式距离或曼哈顿距离:

欧式距离：<img src="https://latex.codecogs.com/gif.latex?d(x,y)=\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}"  /> 

曼哈顿距离: <img src="https://latex.codecogs.com/gif.latex?d(x,y)=\sqrt{\sum_{i=1}^{n}|x_i-y_i|}" title="\sqrt{\sum_{i=1}^{n}|x_i-y_i|}" />
---
title: MLE 极大似然估计
date: 2019-03-27 19:03:27
tags: ["Algorithm"]
categories: Technic
---

## 极大似然估计

MLE即maximum likelihood estimation，极大似然估计，它描述的是这样一种状况：当一件事A发生了，我们认为这件事A（而不是另外一些事B,C,D,...）之所以发生一定是因为这件事A发生的概率比其它所有的事情发生的概率都大。从这个角度，我们可以反推到底是什么前提条件才会使得这件事A发生的概率最大，这个“前提条件”就是我们要求解的。

极大似然估计的结果不一定是符合事实的，但是它确实在现有信息量下面进行的最有可能符合事实的推断。

举个例子：平常不点名的老师在最后一节课点名了，而我恰好不在，老师并不认为这是巧合，而认为我就是全班逃课最勤的那个人。因为只有“我是全班逃课最勤的”，才能导致最大程度地导致“老师点名发现我不在”。老师的思维方式就是“极大似然估计”

## 似然函数

假如从一个装了黑白球的箱子里抓取球，记下颜色再放回去。抓取10次，出现7个黑球3个白球。问箱子里黑白球的比例是多少？

设黑球占比为<img src="https://latex.codecogs.com/gif.latex?\theta"  />。我们知道结果是已经是“7个黑球3个白球”，也知道“7个黑球3个白球的结果”的概率模型是<img src="https://latex.codecogs.com/gif.latex?\theta&space;^{7}(1-\theta)^{3}" />，而需要通过极大似然估计求<img src="https://latex.codecogs.com/gif.latex?\theta" />，也就是需要求使得<img src="https://latex.codecogs.com/gif.latex?\theta&space;^{7}(1-\theta)^{3}" />>最大的<img src="https://latex.codecogs.com/gif.latex?\theta" />值。

所以，极大似然估计的已知信息为：已经发生的事件结果，该事件发生的概率模型结构。而待求解的未知信息为：该模型中的具体参数值
我们将待求参数向该模型的映射称作似然函数，也就是<img src="https://latex.codecogs.com/gif.latex?f(\theta)=\theta&space;^{7}(1-\theta)^{3}" />。求似然函数的极大值，就是求解极大似然估计的过程。此例中，可以求得<img src="https://latex.codecogs.com/gif.latex?\theta=70\%" />时，似然函数取极值。所以答案是70%


## 贝叶斯理论 V.S. 极大似然估计
它们是逆向关系：

1）贝叶斯是已知前提条件是什么，去计算事件A发生的概率

2）极大似然估计是已知事件A发生了，去估计前提条件是什么

## 逻辑回归中的极大似然估计

见[Logistic Regression](https://dorianzi.github.io/2019/03/28/Logistic-Regression/)

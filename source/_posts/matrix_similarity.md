---
title: Matrix Similarity
date: 2019-03-04 00:00:00
tags: "Algorithm"
---

# 相似矩阵

相似矩阵表达的是：同一个线性变换在不同基下的表达。
下面的推导中，A矩阵和B矩阵相似，也就是说他们是同一个线性变换在不同基下的表达


## 推导
见下图


<img src="https://github.com/DorianZi/algorithm_explained/raw/master/res/matrix_similarity.jpg">

首先我们脱离任何基，在“绝对”坐标系里面描述一个线性变换：线性变换将向量<img src="https://latex.codecogs.com/gif.latex?\underset{\alpha}{\rightarrow}" title="\underset{a}{\rightarrow}" style="display:inline;vertical-align:text-top;"/>变换为<img src="https://latex.codecogs.com/gif.latex?\underset{\beta}{\rightarrow}" title="\underset{\beta}{\rightarrow}" style="display:inline;vertical-align:text-top;"/>

接下来我们分别在不同的基里描述这个线性变换：


在基<img src="https://latex.codecogs.com/gif.latex?(\underset{i}{\rightarrow}&space;,\underset{j}{\rightarrow})" title="(\underset{i}{\rightarrow} ,\underset{j}{\rightarrow})" style="display:inline;vertical-align:text-top;"/>下，上述变换描述为：线性变换A将<img src="https://latex.codecogs.com/gif.latex?\underset{v}{\rightarrow}" title="\underset{v}{\rightarrow}" style="display:inline;vertical-align:text-top;">变换为<img src="https://latex.codecogs.com/gif.latex?A\underset{v}{\rightarrow}" title="A\underset{v}{\rightarrow}" style="display:inline;vertical-align:text-top;">

同时由于我们知道向量<img src="https://latex.codecogs.com/gif.latex?\underset{a}{\rightarrow}" title="\underset{a}{\rightarrow}" style="display:inline;vertical-align:text-top;"/>和<img src="https://latex.codecogs.com/gif.latex?\underset{v}{\rightarrow}" title="\underset{v}{\rightarrow}" style="display:inline;vertical-align:text-top;"/>的关系是：<img src="https://latex.codecogs.com/gif.latex?(\underset{i}{\rightarrow},\underset{j}{\rightarrow})\underset{a}{\rightarrow}=\underset{v}{\rightarrow}" title="(\underset{i}{\rightarrow},\underset{j}{\rightarrow})\underset{a}{\rightarrow}=\underset{v}{\rightarrow}" style="display:inline;vertical-align:text-top;">

所以该变换可以进一步表述为：线性变换A将<img src="https://latex.codecogs.com/gif.latex?(\underset{i}{\rightarrow},\underset{j}{\rightarrow})\underset{a}{\rightarrow}" title="(\underset{i}{\rightarrow},\underset{j}{\rightarrow})\underset{a}{\rightarrow}" style="display:inline;vertical-align:text-top;">变换为<img src="https://latex.codecogs.com/gif.latex?A(\underset{i}{\rightarrow},\underset{j}{\rightarrow})\underset{\alpha}{\rightarrow}" title="A(\underset{i}{\rightarrow},\underset{j}{\rightarrow})\underset{\alpha}{\rightarrow}" style="display:inline;vertical-align:text-top;"/>


同理在新基下<img src="https://latex.codecogs.com/gif.latex?(\underset{i'}{\rightarrow}&space;,\underset{j'}{\rightarrow})" title="(\underset{i'}{\rightarrow} ,\underset{j'}{\rightarrow})" style="display:inline;vertical-align:text-top;"/>该变换表述为：线性变换B将<img src="https://latex.codecogs.com/gif.latex?(\underset{i'}{\rightarrow},\underset{j'}{\rightarrow})\underset{\alpha}{\rightarrow}" title="(\underset{i'}{\rightarrow},\underset{j'}{\rightarrow})\underset{\alpha}{\rightarrow}" style="display:inline;vertical-align:text-top;"/>变换为<img src="https://latex.codecogs.com/gif.latex?B(\underset{i'}{\rightarrow},\underset{j'}{\rightarrow})\underset{\alpha}{\rightarrow}" title="B(\underset{i'}{\rightarrow},\underset{j'}{\rightarrow})\underset{\alpha}{\rightarrow}" style="display:inline;vertical-align:text-top;"/>

如果我们知道两组基的转换关系为<img src="https://latex.codecogs.com/gif.latex?P^{-1}" title="P^{-1}" style="display:inline;vertical-align:text-top;"/>，那么有：

<img src="https://latex.codecogs.com/gif.latex?P^{-1}(\underset{i}{\rightarrow},\underset{j}{\rightarrow})=(\underset{i'}{\rightarrow},\underset{j'}{\rightarrow})" title="P^{-1}(\underset{i}{\rightarrow},\underset{j}{\rightarrow})=(\underset{i'}{\rightarrow},\underset{j'}{\rightarrow})" style="display:inline;vertical-align:text-top;"/> ①

同时，线性变换之后得到的向量也应该满足基之间的转换关系：

<img src="https://latex.codecogs.com/gif.latex?P^{-1}A(\underset{i}{\rightarrow},\underset{j}{\rightarrow})\underset{\alpha}{\rightarrow}=B(\underset{i'}{\rightarrow},\underset{j'}{\rightarrow})\underset{\alpha}{\rightarrow}" title="P^{-1}A(\underset{i}{\rightarrow},\underset{i}{\rightarrow})\underset{\alpha}{\rightarrow}=B(\underset{i'}{\rightarrow},\underset{j'}{\rightarrow})\underset{\alpha}{\rightarrow}" style="display:inline;vertical-align:text-top;"/>  ②

将①带入②得到：

<img src="https://latex.codecogs.com/gif.latex?P^{-1}A=BP^{-1}" title="P^{-1}A=BP^{-1}" style="display:inline;vertical-align:text-top;"/>

==>

<img src="https://latex.codecogs.com/gif.latex?P^{-1}AP=B" title="P^{-1}AP=B" style="display:inline;vertical-align:text-top;"/>



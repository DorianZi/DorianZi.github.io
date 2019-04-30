---
title: Random Forest 随机森林
date: 2019-04-17 21:31:03
tags: ["Algorithm"]
categories: Technic
---

Random Forest 随机森林，它的名字包含了两个信息 1）是用很多[决策树](https://dorianzi.github.io/2019/03/29/Decision-Tree/)组成的森林；2）是采用随机的特征选取和样本选取方法。

它的名字没有包含的信息或许更加重要：Bagging方法

# Bagging

Bagging 常常和Boosting放一起讨论，他们作为Ensemble Learning集成学习的两种方法而存在。

Bagging指的是有放回的方法，在随机森林的构建上就是：每棵决策树用的是同样的样本集（当然，每次真正选择的参与建树的样本可以不同，以减轻over-fitting的可能）。也就是说这些样本在用来构建一棵树之后，又放回去了，再来构建后面的树。所以与Boosting加法性质不同，Bagging构建出来的每棵树之间是平行关系。那么最后决策采用<font size="4">投票</font>方式，就比较容易理解了。

![](/uploads/randforest_1.png)

# 每棵树的生成

如果训练集为N,特征数为M,则一棵树的生成过程如下：

1）随机抽取<img src="https://latex.codecogs.com/gif.latex?n<N"  />个样本

2）随机抽取<img src="https://latex.codecogs.com/gif.latex?m<M"  />个特征

3）使用<img src="https://latex.codecogs.com/gif.latex?n\times&space;m" title="n\times m" />子集，构建决策树（ID3,C4.5,CART）。以CART分类树为例，创建任何一棵树的过程：

3.1）选择m个特征分裂里[基尼指数](https://dorianzi.github.io/2019/03/30/CART/#%E5%9F%BA%E5%B0%BC%E6%8C%87%E6%95%B0)最高的那个，开始二叉（CART分类树使用的是二叉）分裂

3.2）如果到了叶子节点，则该分支停止，否则继续选择基尼指数最高的特征继续分裂

3.3）如果分裂用尽了所有特征，而还没到叶子节点(通常是因为最初m个随机特征的抽取的时候，某些决定性特征没有被选中导致的)，则以某个类别的样本数量最多的为强制叶子节点

4）创建N棵树(N为停止条件，试情况而定)之后，停止建树，采用投票机制选择得票最高的那个类别

以上为训练过程，预测过程就是用待预测数据走通上述过程，得到预测结果投票最高的那个。

# 随机森林举例

以CART里讲过的[数据集](https://dorianzi.github.io/2019/03/30/CART/#%E5%88%86%E7%B1%BB%E6%A0%91%E7%89%B9%E5%BE%81%E5%88%86%E8%A3%82)为例，预测是否放贷：

![](/uploads/decision_tree_1.png)

1）随机抽取9个样本

2）随机抽取三个特征：是否青年，是否有工作，是否有自己房子

样本，特征选取如下：

![](/uploads/randforest_2.png)

以<img src="https://latex.codecogs.com/gif.latex?A_1=1,A_2=1，A_3=1" title="A_1=1,A_2=1，A_3=1" />>分别表示青年，有工作，有房子三个特征，则Gini指数分别为：

<img src="https://latex.codecogs.com/gif.latex?Gini(D,A_1=1)=\frac{4}{9}Gini(D_1)&plus;\frac{5}{9}Gini(D_2)=\frac{4}{9}\times&space;\frac{2}{4}&plus;\frac{5}{9}\times\frac{12}{25}=0.224" title="Gini(D,A_1=1)=\frac{4}{9}Gini(D_1)+\frac{5}{9}Gini(D_2)=\frac{4}{9}\times \frac{2}{4}+\frac{5}{9}\times\frac{12}{25}=0.224" />

<img src="https://latex.codecogs.com/gif.latex?Gini(D,A_2=1)=\frac{4}{9}Gini(D_1)&plus;\frac{5}{9}Gini(D_2)=\frac{4}{9}\times&space;0&plus;\frac{5}{9}\times\frac{8}{25}=0.178" title="Gini(D,A_1=1)=\frac{4}{9}Gini(D_1)+\frac{5}{9}Gini(D_2)=\frac{4}{9}\times 0+\frac{5}{9}\times\frac{8}{25}=0.178" />

<img src="https://latex.codecogs.com/gif.latex?Gini(D,A_3=1)=\frac{3}{9}Gini(D_1)&plus;\frac{6}{9}Gini(D_2)=\frac{3}{9}\times&space;0&plus;\frac{6}{9}\times\frac{4}{9}=0.296" title="Gini(D,A_1=1)=\frac{4}{9}Gini(D_1)+\frac{5}{9}Gini(D_2)=\frac{3}{9}\times 0+\frac{6}{9}\times\frac{4}{9}=0.296" />

其中“有工作”与否的分裂点基尼指数最小，所以以它为分裂点进行分裂

![](/uploads/randforest_3.png)

接下来计算没工作情况下的分支，数据为：

![](/uploads/randforest_4.png)

计算青年，有房子两个特征的基尼指数：

<img src="https://latex.codecogs.com/gif.latex?Gini(D,A_1=1)=\frac{2}{5}Gini(D_1)&plus;\frac{3}{5}Gini(D_2)=\frac{2}{5}\times&space;0&plus;\frac{3}{5}\times\frac{4}{9}=0.267" title="Gini(D,A_1=1)=\frac{2}{5}Gini(D_1)+\frac{3}{5}Gini(D_2)=\frac{2}{5}\times 0+\frac{3}{5}\times\frac{4}{9}=0.267" />

<img src="https://latex.codecogs.com/gif.latex?Gini(D,A_1=1)=\frac{1}{5}Gini(D_1)&plus;\frac{4}{5}Gini(D_2)=\frac{1}{5}\times&space;0&plus;\frac{4}{5}\times&space;0=0" title="Gini(D,A_1=1)=\frac{1}{5}Gini(D_1)+\frac{4}{5}Gini(D_2)=\frac{1}{5}\times 0+\frac{4}{5}\times 0=0" />

“有房子”与否的基尼指数最小，选择它。有：

<font size="4" color="Blue">第1棵树:</font>
![](/uploads/randforest_5.png)

第一棵树建树完毕。

3）重复 1），2）建更多的树（不妨设总棵数为4）：

<font size="4" color="Blue">第2棵树:</font>
![](/uploads/randforest_6.png)

<font size="4" color="Blue">第3棵树:</font>
![](/uploads/randforest_7.png)

<font size="4" color="Blue">第4棵树:</font>
![](/uploads/randforest_8.png)

4）现在有一个人的特征是：中年，没工作，没房子，信贷非常好。 预测是否向Ta放贷？

第1棵树：不放贷
第2棵树：不放贷
第3棵树：不放贷
第4棵树：放贷

通过投票，放贷票数最多。所以最后预测结果为：不放贷。


以上。
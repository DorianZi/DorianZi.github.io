---
title: CART 分类与回归树
date: 2019-03-30 18:07:53
tags: ["Algorithm"]
categories: Technic
---

# CART定义
在上一篇[决策树](https://dorianzi.github.io/2019/03/29/Decision-Tree)中，我们已经介绍了决策树的原理和ID3,C4.5算法。这一篇专门讲解CART算法。

ART即Classification and Regression Tree,分类与回归树。特点是：只有二叉树。

## 分类树
CART用作分类树的时候跟ID3,C4.5决策树算法类似，只是特征分裂的准则不一样。 CART首先是二叉树，其次分裂评价标准是用基尼指数

### 基尼指数
基尼指数可以用来描述一个集合D的类别纯度，集合的样本们只属于一类，那该集合的基尼指数为0，集合的样本属于越多类，则基尼指数越大。基尼指数对集合的算法是：

<img src="https://latex.codecogs.com/gif.latex?Gini(D&space;)=\sum_{k=1}^{K}\rho&space;_k(1-\rho&space;_k)=1-\sum_{k=1}^{K}\rho&space;_k^2" />

该式子的解读：集合的样本属于K个类，现在随机抽取一个样本，它属于第k个类的概率为<img src="https://latex.codecogs.com/gif.latex?\rho&space;_k"  />。设集合里一共有n个样本，第k个类的样本有m个，则<img src="https://latex.codecogs.com/gif.latex?\rho&space;_k=\frac{m}{n}" />

在特征A取值为a的条件下的集合D的基尼指数为：

<img src="https://latex.codecogs.com/gif.latex?Gini(D,A)=\frac{|D_1|}{|D|}Gini(D_1)&plus;\frac{|D_2|}{|D|}Gini(D_2)" />

上面式子中，<img src="https://latex.codecogs.com/gif.latex?D_1"  />为A取a时的样本集合，<img src="https://latex.codecogs.com/gif.latex?D_2" title="D_2" />为其它样本的集合：<img src="https://latex.codecogs.com/gif.latex?D_2=D-D_1" title="D_2=D-D_1" />

可以看到特征A取值之后，影响了集合<img src="https://latex.codecogs.com/gif.latex?D" />的密度，形成了<img src="https://latex.codecogs.com/gif.latex?D_1" />和<img src="https://latex.codecogs.com/gif.latex?D_2"  />两部分的密度，所以需要加权。

### 分类树特征分裂
我们依然考虑此数据集：

![](/uploads/decision_tree_1.png)

设<img src="https://latex.codecogs.com/gif.latex?A_1,A_2,A_3,A_4" title="A_1,A_2,A_3,A_4" />表示年龄段，有工作与否，有自己房子，信贷情况4个特征。<img src="https://latex.codecogs.com/gif.latex?A_1=1,2,3" />表示青年，中年，老年；<img src="https://latex.codecogs.com/gif.latex?A_2=1,2"  />表示有工作和没工作；<img src="https://latex.codecogs.com/gif.latex?A_3=1,2"  />表示有房子和没房子；<img src="https://latex.codecogs.com/gif.latex?A_4=1,2,3"  />表示信贷非常好，好，一般。

1）假设以“青年”与否为分裂点，即<img src="https://latex.codecogs.com/gif.latex?A_1=1"/>，求基尼指数。

<img src="https://latex.codecogs.com/gif.latex?Gini(D,A_1=1)=\frac{5}{15}Gini(D_1)&plus;\frac{10}{15}Gini(D_2)=\frac{5}{15}\times&space;\frac{12}{25}&plus;\frac{10}{15}\times&space;\frac{21}{50}=0.44"  />


2）假设以“中年”与否为分裂点，即<img src="https://latex.codecogs.com/gif.latex?A_1=2"/>，求基尼指数。

<img src="https://latex.codecogs.com/gif.latex?Gini(D,A_1=2)}=0.48"  />

3）假设以“老年”与否为分裂点，即<img src="https://latex.codecogs.com/gif.latex?A_1=3"/>，求基尼指数。

<img src="https://latex.codecogs.com/gif.latex?Gini(D,A_1=3)}=0.44"  />

4）假设以“有工作”与否为分裂点，即<img src="https://latex.codecogs.com/gif.latex?A_2=1"/>或<img src="https://latex.codecogs.com/gif.latex?A_2=2"/>，求基尼指数（只有两种情况，基尼指数是相等的）。

<img src="https://latex.codecogs.com/gif.latex?Gini(D,A_2=1)}=0.32"  />

5）假设以“有房子与否”与否为分裂点，即<img src="https://latex.codecogs.com/gif.latex?A_3=1"/>或<img src="https://latex.codecogs.com/gif.latex?A_3=2"/>，求基尼指数。

<img src="https://latex.codecogs.com/gif.latex?Gini(D,A_3=1)}=0.27"  />

6）假设以“信贷情况非常好”与否为分裂点，即<img src="https://latex.codecogs.com/gif.latex?A_4=1"/>，求基尼指数。

<img src="https://latex.codecogs.com/gif.latex?Gini(D,A_4=1)==0.36"  />


7）假设以“信贷情况好”与否为分裂点，即<img src="https://latex.codecogs.com/gif.latex?A_4=2"/>，求基尼指数。

<img src="https://latex.codecogs.com/gif.latex?Gini(D,A_4=2)}=0.47"  />

8）假设以“信贷情况一般”与否为分裂点，即<img src="https://latex.codecogs.com/gif.latex?A_4=3"/>，求基尼指数。

<img src="https://latex.codecogs.com/gif.latex?Gini(D,A_4=3)}=0.32"  />

9）比较以上所有的Gini指数，<img src="https://latex.codecogs.com/gif.latex?Gini(D,A_3=1)}=0.27"  />最小，所以<font size="4">“有房子”</font>与否为最优分裂点

10) 分裂之后的决策图如下：

![](/uploads/decision_tree_3.png)

接下来我们继续寻找分裂点进行分裂

目前数据集合被缩小为：

![](/uploads/decision_tree_4.png)

11）以此时的集合为基础，循环到1)开始，计算除了“有房子”与否以外的特征作为分裂点的基尼指数。

中间计算过程省略，直接给出结果：<img src="https://latex.codecogs.com/gif.latex?A_2=1"  />时最小，所以<font size="4">“有工作”</font>与否为最优分裂点。

继续绘制决策图：

![](/uploads/decision_tree_5.png)

决策完毕。

## 回归树
回归树结合了线性回归和决策树，本质上是决策树，实现的是分段回归。

对图示的数据点，线性回归为<font color="Red">红色</font>，回归树为<font color="Green">绿色</font>

![](/uploads/CART_1.png)

下面开始讲解回归树原理

### 回归树算法原理

首先对于特征进行二叉特征分裂，先尝试所有分裂点，再通过最小损失函数决定哪个分裂点是最佳的，并进行实质分裂。分裂出来两条分支，每个分支为一个子数据集（这就是为什么上图x坐标是分段的），每个子数据集上的样本的结果都等于它们的平均值（这就是为什么上图y坐标每一段都是同一个值）。然后在每一个子数据集重复上面的的分裂，直到停止条件。 停止条件是人为决定的，比如决定只分裂到第3层。


### 回归树损失函数
其中的损失函数通过最小二乘法获得，即当同一个点分裂为两个子数据集后，每个子数据集的最小二乘误差最小化，也就是两个子数据集的最小二乘误差的和要最小化。数学表达如下：

<img src="https://latex.codecogs.com/gif.latex?\underset{j,s}{min}[\underset{c_1}{min}\sum_{x_i\epsilon&space;R_1(j,s)}^{&space;}(y_i-c_1)^2&plus;\underset{c_2}{min}\sum_{x_i\epsilon&space;R_2(j,s)}^{&space;}(y_i-c_2)^2]"  />

上式中<img src="https://latex.codecogs.com/gif.latex?c_1,c_2" title="c_1,c_2" />为两个子集中计算结果值（即平均值），<img src="https://latex.codecogs.com/gif.latex?y_i" title="y_i" />为实际结果值

### 回归树实例分析
考虑以下数据集，x为特征变量，y为结果

![](/uploads/CART_2.png)

绘图为：

![](/uploads/CART_4.png)

1）只有一个特征变量x,故对它进行分裂。

考虑这9个分裂点<img src="https://latex.codecogs.com/gif.latex?[1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5]"/> ，之所以不使用1,2,3这种实际特征取值为分裂点，是为了划分子集的清晰。

2） 假设分裂点为<img src="https://latex.codecogs.com/gif.latex?s=1.5" title="s=1.5" />，获得两个子集：
![](/uploads/CART_3.png)

计算子集<img src="https://latex.codecogs.com/gif.latex?R_1" title="R_1" />的结果：<img src="https://latex.codecogs.com/gif.latex?c_1=(4.5)/1=4.5" title="c_1=(4.5+4.75)/2=4.625" />

计算子集<img src="https://latex.codecogs.com/gif.latex?R_2" title="R_2" />的结果：<img src="https://latex.codecogs.com/gif.latex?c_2=(4.75&plus;4.91&plus;5.34&plus;5.80&plus;7.05&plus;7.90&plus;8.23&plus;8.70&plus;9.00)/9=6.853"  />

计算结果绘制到坐标图中：

![](/uploads/CART_5.png)

计算这次分裂的损失量：

<img src="https://latex.codecogs.com/gif.latex?loss(1.5)=(4.5-4.5)^2&plus;[(4.75-6.853)^2&plus;(4.91-6.853)^2&plus;(5.34-6.853)^2&plus;(5.80-6.853)^2&plus;(7.05-6.853)^2&plus;(7.90-6.853)^2&plus;(8.23-6.853)^2&plus;(8.70-6.853)^2&plus;(9.00-6.853)^2]=22.648"  />

3） 假设分裂点为<img src="https://latex.codecogs.com/gif.latex?s=2.5" />，获得两个子集：
![](/uploads/CART_6.png)

计算子集<img src="https://latex.codecogs.com/gif.latex?R_1" title="R_1" />的结果：<img src="https://latex.codecogs.com/gif.latex?c_1=(4.5&plus;4.75)/2=4.625" />

计算子集<img src="https://latex.codecogs.com/gif.latex?R_2" title="R_2" />的结果：<img src="https://latex.codecogs.com/gif.latex?c_2=(4.91&plus;5.34&plus;5.80&plus;7.05&plus;7.90&plus;8.23&plus;8.70&plus;9.00)/8=7.116"  />

计算结果绘制到坐标图中：

![](/uploads/CART_7.png)

计算这次分裂的损失量：

<img src="https://latex.codecogs.com/gif.latex?loss(2.5)=[(4.5-4.625)^2&plus;(4.75-4.625)^2]&plus;[(4.91-7.116)^2&plus;(5.34-7.116)^2&plus;(5.80-7.116)^2&plus;(7.05-7.116)^2&plus;(7.90-7.116)^2&plus;(8.23-7.116)^2&plus;(8.70-7.116)^2&plus;(9.00-7.116)^2]=17.702"/>

4）继续计算分裂点<img src="https://latex.codecogs.com/gif.latex?s=3.5,4.5,5.5,6.5,7.5,8.5,9.5"  />时候的结果，得到所有结果为下表：

![](/uploads/CART_8.png)

比较所有损失，当s=5.5的时候分裂，损失最少，所以从<font size="4">s=5.5</font>进行第一次分裂，分裂之后的坐标图为：

![](/uploads/CART_9.png)

5）至此，数据被分裂为两个子集：

![](/uploads/CART_10.png)

对<img src="https://latex.codecogs.com/gif.latex?D_1" title="D_1" />进行1)~4)的分裂过程，得到：

![](/uploads/CART_11.png)

当s=3.5的时候分裂，损失最少，所以从<font size="4">s=3.5</font>对<img src="https://latex.codecogs.com/gif.latex?D_1" title="D_1" />进行分裂

对<img src="https://latex.codecogs.com/gif.latex?D_2" title="D_2" />进行1)~4)的分裂过程，得到：

![](/uploads/CART_12.png)

当s=7.5的时候分裂，损失最少，所以从<font size="4">s=7.5</font>对<img src="https://latex.codecogs.com/gif.latex?D_2" title="D_2" />进行分裂

绘制坐标图

![](/uploads/CART_13.png)

我们看一下此刻的树图：

![](/uploads/CART_14.png)

如果觉得目前的回归精度满足要求（比如规定一个子集里面的样本数低于事先设定的阈值），可以就此时终止，则我们的回归树已经构建了3层了（称深度为3。
不过，为了精度再高一点，我们再构建一层。

6）第三层构建开始，目前4个子集为：

![](/uploads/CART_15.png)

对4个子集<img src="https://latex.codecogs.com/gif.latex?D11,D12,D21,D22" title="D11,D12,D21,D22" />分别进行1）~ 4）的循环。省略中间过程，直接给结果：

![](/uploads/CART_16.png)

D11: 选择s=1.5分裂
D12: 选择s=4.5分裂
D21: 选择s=6.5分裂
D22: 选择s=8.5分裂

绘制坐标图：

![](/uploads/CART_17.png)

树图：

![](/uploads/CART_18.png)

构建完毕。

BTW,事实上D11和D21已经过拟合，因为它本身只有两个边界点，从结果上看loss也已经为0了

## 剪枝算法

### 预剪枝
在构建树的过程中依照指定的标准而停止构建某个分支，常用的方法有：

1）定义一个深度，当树到达该深度就停止，如上面用到的
2）定义一个阈值，当某个节点子集的样本数小于该阈值，就停止
3）定义一个阈值，计算每次扩张对系统性能的增益，当增益值小于该阈值，就停止

### 后剪枝

先构建完整的树（允许过拟合），然后剪掉置信度不够的子树，并用子树种最频繁的类别或值代替

![](/uploads/CART_19.png)

剪枝的方法有：

Reduced-Error Pruning(REP,错误率降低剪枝）

Pesimistic-Error Pruning(PEP,悲观错误剪枝）

Cost-Complexity Pruning（CCP，代价复杂度剪枝)

EBP(Error-Based Pruning)（基于错误的剪枝）


# 参考
https://www.cnblogs.com/starfire86/p/5749334.html


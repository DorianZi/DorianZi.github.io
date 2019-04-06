---
title: Decision Tree 决策树
date: 2019-03-29 17:30:39
tags: ["Algorithm"]
categories: Technic
---

## 决策树

### 决策树的定义

决策树类似于人们进行决策的过程，从一个根节点开始，通过逐一特征的筛选，生成多个分支，每个叶子节点为一个分类

考虑下面的样本：

![](/uploads/decision_tree_1.png)

人们的决策过程：

![](/uploads/decision_tree_2.png)

我们看到对于青年来说，只要工作情况就能决定是否房贷，房子和信贷情况并没有起到区分作用。对所有年龄层的来说，信贷情况都是多余的特征。这个决策过程看起来不错，不过这就是最佳路径了吗？

而在实际操作中，我们是用什么算法策略决定每一步特征的选取呢？也就是说，我们进行特征选择是基于什么准则？

### 特征选择

我们基于信息增益来选择特征。信息增益指的是信息熵的增益，为理解它需要提前了解两个概念：信息熵，条件熵

#### 信息熵

一个事件发生的概率越低，它的不确定性就越强，信息量就越大，所以我们要用概率的函数来表示信息量,即<img src="https://latex.codecogs.com/gif.latex?f(P(A))" />；多个独立事件发生的概率是它们各自概率的乘积，而信息量是他们各自信息量的和,即

<img src="https://latex.codecogs.com/gif.latex?f(P(A_1))&plus;f(P(A_2))&plus;...&plus;f(P(A_n))=f(P(A_1)P(A_2)...P(A_n))"  />

满足这个条件的函数是对数，于是用对数表示信息量：

<img src="https://latex.codecogs.com/gif.latex?f(P(A))=-log\rho "  />

其中<img src="https://latex.codecogs.com/gif.latex?\rho&space;=P(A)" title="\rho =P(A)" />

上面说的是一个具体的事件或特征。对一个变量X来说，它可能取不同的值，它取任何一个值都是一个事件，且所有取值的事件互相独立。我们考虑X的信息就应该考虑它所有取值，那么合理的方式就是使用它所有取值信息量的期望：

<img src="https://latex.codecogs.com/gif.latex?H(X)=P(X_1)f(P(X_1))&plus;P(X_2)f(P(X_2))&plus;...&plus;P(X_n)f(P(X_n))"  />

<img src="https://latex.codecogs.com/gif.latex?=\rho_1&space;(-log\rho_1&space;)&plus;\rho_2&space;(-log\rho_2&space;)&plus;...&plus;\rho_n&space;(-log\rho_n&space;)"  />

<img src="https://latex.codecogs.com/gif.latex?=-\sum_{i=1}^{n}\rho&space;_ilog\rho&space;_i"  />

我们将此称作信息熵。所以说信息熵指的是变量的属性。

#### 条件熵
现在在X基础上再加入另一个变量Y，它们的联合概率分布（与条件概率分别的区别见附录）为

<img src="https://latex.codecogs.com/gif.latex?P(X=x_i,Y=y_j)=\rho&space;_{ij},\&space;\&space;\&space;\&space;\&space;\&space;i=1,2,...,n;\&space;j=1,2,...,m"  />

定义条件熵为Y在X条件下的熵：

<img src="https://latex.codecogs.com/gif.latex?H(Y|X)=\sum_{i=1}^{n}\rho&space;_iH(Y|X=x_i)" title="H(Y|X)=\sum_{i=1}^{n}\rho _iH(Y|X=x_i)" />

其中<img src="https://latex.codecogs.com/gif.latex?\rho_i&space;=P(X=x_i)"  />

观察条件熵的定义公式，从外往里看，它是X取值变动下对熵<img src="https://latex.codecogs.com/gif.latex?H(Y|X=x_i)" />的加权和，而本身<img src="https://latex.codecogs.com/gif.latex?H(Y|X=x_i)" />是固定<img src="https://latex.codecogs.com/gif.latex?X=x_i" />下的，Y取值变动下每个条件概率<img src="https://latex.codecogs.com/gif.latex?P(Y=y_j|X=x_i)" />的加权和。所以它是期望的期望，或者说是熵的熵，是信息量的嵌套加权。

推导一下条件熵具体的计算方法：

<img src="https://latex.codecogs.com/gif.latex?H(Y|X=x_i)=-\sum_{j=1}^{m}P(Y=y_j|X=x_i)logP(Y=y_j|X=x_i)"  />

<img src="https://latex.codecogs.com/gif.latex?=-\sum_{j=1}^{m}\frac{P(X=x_i,Y=y_j)}{P(X=x_i)}log\frac{P(X=x_i,Y=y_j)}{P(X=x_i)}"  />

<img src="https://latex.codecogs.com/gif.latex?=-\sum_{j=1}^{m}\frac{\rho&space;_{ij}}{\rho&space;_{i}}log\frac{\rho&space;_{ij}}{\rho&space;_{i}}" />

则：

<img src="https://latex.codecogs.com/gif.latex?H(Y|X)=-\sum_{i=1}^{n}\rho&space;_i\sum_{j=1}^{m}\frac{\rho&space;_{ij}}{\rho&space;_i}log\frac{\rho&space;_{ij}}{\rho&space;_i}"  />

<img src="https://latex.codecogs.com/gif.latex?=-\sum_{i=1}^{n}\sum_{j=1}^{m}\rho&space;_{ij}log\frac{\rho&space;_{ij}}{\rho&space;_i}"  />

<img src="https://latex.codecogs.com/gif.latex?=-\sum_{i=1}^{n}\sum_{j=1}^{m}P(X=x_i,Y=y_j)log\frac{P(X=x_i,Y=y_j)}{P(X=x_i)}" />&emsp;&emsp;&emsp;&emsp;①

<img src="https://latex.codecogs.com/gif.latex?=-\sum_{i=1}^{n}\sum_{j=1}^{m}P(X=x_i,Y=y_j)logP(Y=y_j|X=x_i)"  />&emsp;&emsp;&emsp;&emsp;②

由②可见条件熵由联合概率和条件概率构成。由①可见，条件概率可以由联合概率和作为条件的事件概率求得

#### 信息增益

了解了信息熵（或者称经验熵）和条件熵，可以来理解信息增益了。

信息增益的定义是，当一个特征A引入后，原来集合D的经验熵，变成了D对A的条件熵，这个变化过程熵减少了多少。用数学表示：

<img src="https://latex.codecogs.com/gif.latex?g(D,A)=H(D)-H(D|A)"  />

BTW，因为特征在不断引入，每次引入之后的条件熵就成为下一次特征引入前的经验熵。

决策树的特征选择基于信息增益最大，也就是新特征的加入使得整个集合的信息量减少最多，概率就越大，这样就越接近最终分类结果。

#### 基于信息增益进行特征选择

现在开始通过计算信息增益来对上述样本进行特征选择。

1.1）首先计算初始经验熵，初始只有一个特征D，就是放贷与否。放贷为<img src="https://latex.codecogs.com/gif.latex?\frac{9}{15}"/>，不放为<img src="https://latex.codecogs.com/gif.latex?\frac{6}{15}"  />，则经验熵为：

<img src="https://latex.codecogs.com/gif.latex?H(D)=-\frac{9}{15}log_2\frac{9}{15}-\frac{6}{15}log_2\frac{6}{15}=0.971"  />

1.2）假设引入<font size="5">“年龄段”</font>特征。则原来对样本的了解只有放贷与否，现在多了一个“年龄段”的区分度。设“年龄段”为<img src="https://latex.codecogs.com/gif.latex?A_1" />,使用上述②式，计算条件熵：

P(青年，放贷)=<img src="https://latex.codecogs.com/gif.latex?\frac{2}{15}"  />&emsp;&emsp;&emsp;P(放贷|青年)=<img src="https://latex.codecogs.com/gif.latex?\frac{2}{5}"  />&emsp;&emsp;&emsp;P(青年，不放贷)=<img src="https://latex.codecogs.com/gif.latex?\frac{3}{15}"  />&emsp;&emsp;&emsp;P(不放贷|青年)=<img src="https://latex.codecogs.com/gif.latex?\frac{3}{5}"  />

P(中年，放贷)=<img src="https://latex.codecogs.com/gif.latex?\frac{3}{15}"  />&emsp;&emsp;&emsp;P(放贷|中年)=<img src="https://latex.codecogs.com/gif.latex?\frac{3}{5}"  />&emsp;&emsp;&emsp;P(中年，不放贷)=<img src="https://latex.codecogs.com/gif.latex?\frac{2}{15}"  />&emsp;&emsp;&emsp;P(不放贷|中年)=<img src="https://latex.codecogs.com/gif.latex?\frac{2}{5}"  />

P(老年，放贷)=<img src="https://latex.codecogs.com/gif.latex?\frac{4}{15}"  />&emsp;&emsp;&emsp;P(放贷|老年)=<img src="https://latex.codecogs.com/gif.latex?\frac{4}{5}"  />&emsp;&emsp;&emsp;P(老年，不放贷)=<img src="https://latex.codecogs.com/gif.latex?\frac{1}{15}"  />&emsp;&emsp;&emsp;P(不放贷|老年)=<img src="https://latex.codecogs.com/gif.latex?\frac{1}{5}"  />

<img src="https://latex.codecogs.com/gif.latex?H(D|A_1)=-(\frac{2}{15}log_2\frac{2}{5}&plus;\frac{3}{15}log_2\frac{3}{5})-(\frac{3}{15}log_2\frac{3}{5}&plus;\frac{2}{15}log_2\frac{2}{5})-(\frac{4}{15}log_2\frac{4}{5}&plus;\frac{1}{15}log_2\frac{1}{5})=0.888" />

=>

<img src="https://latex.codecogs.com/gif.latex?g(D,A_1)=H(D)-H(D|A_1)=0.971-0.888=0.083"  />

1.3）假设引入<font size="5">“有工作与否”</font>特征。设“有工作与否”为<img src="https://latex.codecogs.com/gif.latex?A_2" />,使用上述②式，计算条件熵：

P(有工作，放贷)=<img src="https://latex.codecogs.com/gif.latex?\frac{5}{15}"  />&emsp;&emsp;&emsp;P(放贷|有工作)=<img src="https://latex.codecogs.com/gif.latex?\frac{5}{5}"  />&emsp;&emsp;&emsp;P(有工作，不放贷)=<img src="https://latex.codecogs.com/gif.latex?\frac{0}{15}"  />&emsp;&emsp;&emsp;P(不放贷|有工作)=<img src="https://latex.codecogs.com/gif.latex?\frac{0}{5}"  />

P(没工作，放贷)=<img src="https://latex.codecogs.com/gif.latex?\frac{4}{15}"  />&emsp;&emsp;&emsp;P(放贷|没工作)=<img src="https://latex.codecogs.com/gif.latex?\frac{4}{10}"  />&emsp;&emsp;&emsp;P(没工作，不放贷)=<img src="https://latex.codecogs.com/gif.latex?\frac{6}{15}"  />&emsp;&emsp;&emsp;P(不放贷|没工作)=<img src="https://latex.codecogs.com/gif.latex?\frac{6}{10}"  />

<img src="https://latex.codecogs.com/gif.latex?H(D|A_2)=-(\frac{5}{15}log_2\frac{5}{5}&plus;\frac{0}{15}log_2\frac{0}{5})-(\frac{4}{15}log_2\frac{4}{10}&plus;\frac{6}{15}log_2\frac{6}{10})=0.674" />

=>

<img src="https://latex.codecogs.com/gif.latex?g(D,A_2)=H(D)-H(D|A_2)=0.971-0。647=0.324"  />

1.4）假设引入<font size="5">“有房子与否”</font>特征。设“有房子与否”为<img src="https://latex.codecogs.com/gif.latex?A_3" />,使用上述②式，计算条件熵：

P(有房子，放贷)=<img src="https://latex.codecogs.com/gif.latex?\frac{6}{15}"  />&emsp;&emsp;&emsp;P(放贷|有房子)=<img src="https://latex.codecogs.com/gif.latex?\frac{6}{6}"  />&emsp;&emsp;&emsp;P(有房子，不放贷)=<img src="https://latex.codecogs.com/gif.latex?\frac{0}{15}"  />&emsp;&emsp;&emsp;P(不放贷|有房子)=<img src="https://latex.codecogs.com/gif.latex?\frac{0}{6}"  />

P(没房子，放贷)=<img src="https://latex.codecogs.com/gif.latex?\frac{3}{15}"  />&emsp;&emsp;&emsp;P(放贷|没房子)=<img src="https://latex.codecogs.com/gif.latex?\frac{3}{9}"  />&emsp;&emsp;&emsp;P(没房子，不放贷)=<img src="https://latex.codecogs.com/gif.latex?\frac{6}{15}"  />&emsp;&emsp;&emsp;P(不放贷|没房子)=<img src="https://latex.codecogs.com/gif.latex?\frac{6}{9}"  />

<img src="https://latex.codecogs.com/gif.latex?H(D|A_3)=-(\frac{6}{15}log_2\frac{6}{6}&plus;\frac{0}{15}log_2\frac{0}{6})-(\frac{3}{15}log_2\frac{3}{9}&plus;\frac{6}{15}log_2\frac{6}{9})=0.551" />

=>

<img src="https://latex.codecogs.com/gif.latex?g(D,A_3)=H(D)-H(D|A_3)=0.971-0。551=0.420"  />

1.5）假设引入<font size="5">“信贷情况”</font>特征。设“信贷情况”为<img src="https://latex.codecogs.com/gif.latex?A_4" />,使用上述②式，计算条件熵：

P(信贷一般，放贷)=<img src="https://latex.codecogs.com/gif.latex?\frac{1}{15}"  />&emsp;&emsp;&emsp;P(放贷|信贷一般)=<img src="https://latex.codecogs.com/gif.latex?\frac{1}{5}"  />&emsp;&emsp;&emsp;P(信贷一般，不放贷)=<img src="https://latex.codecogs.com/gif.latex?\frac{4}{15}"  />&emsp;&emsp;&emsp;P(不放贷|信贷一般)=<img src="https://latex.codecogs.com/gif.latex?\frac{4}{5}"  />

P(信贷好，放贷)=<img src="https://latex.codecogs.com/gif.latex?\frac{4}{15}"  />&emsp;&emsp;&emsp;P(放贷|信贷好)=<img src="https://latex.codecogs.com/gif.latex?\frac{4}{6}"  />&emsp;&emsp;&emsp;P(信贷好，不放贷)=<img src="https://latex.codecogs.com/gif.latex?\frac{2}{15}"  />&emsp;&emsp;&emsp;P(不放贷|信贷好)=<img src="https://latex.codecogs.com/gif.latex?\frac{2}{6}"  />

P(信贷非常好，放贷)=<img src="https://latex.codecogs.com/gif.latex?\frac{4}{15}"  />&emsp;&emsp;&emsp;P(放贷|信贷非常好)=<img src="https://latex.codecogs.com/gif.latex?\frac{4}{4}"  />&emsp;&emsp;&emsp;P(信贷非常好，不放贷)=<img src="https://latex.codecogs.com/gif.latex?\frac{0}{15}"  />&emsp;&emsp;&emsp;P(不放贷|信贷非常好)=<img src="https://latex.codecogs.com/gif.latex?\frac{0}{4}"  />

<img src="https://latex.codecogs.com/gif.latex?H(D|A_4)=-(\frac{1}{15}log_2\frac{1}{5}&plus;\frac{4}{15}log_2\frac{4}{5})-(\frac{4}{15}log_2\frac{4}{6}&plus;\frac{2}{15}log_2\frac{2}{6})-(\frac{4}{15}log_2\frac{4}{4}&plus;\frac{0}{15}log_2\frac{0}{4})=0.608" />

=>

<img src="https://latex.codecogs.com/gif.latex?g(D,A_4)=H(D)-H(D|A_4)=0.971-0.608=0.363" />

1.6）比较<img src="https://latex.codecogs.com/gif.latex?g(D,A_1)" />，<img src="https://latex.codecogs.com/gif.latex?g(D,A_2)" />，<img src="https://latex.codecogs.com/gif.latex?g(D,A_3)" />，<img src="https://latex.codecogs.com/gif.latex?g(D,A_4)" />&emsp;=>&emsp;<img src="https://latex.codecogs.com/gif.latex?g(D,A_3)>g(D,A_4)>g(D,A_2)>g(D,A_1)"  />

也就是这里我们选择特征<font size="5">“有房子与否”</font>，至此我们做出了第一个特征选择，在选择之后根据有房子与否的“是”与“否”岔开了两个路径,也就是分割出来了两个子空间<img src="https://latex.codecogs.com/gif.latex?D_1" title="D_1" />和<img src="https://latex.codecogs.com/gif.latex?D_2" title="D_1" />，其中“是”空间的路径已经抵达了叶子；“否”空间还有待开发。

![](/uploads/decision_tree_3.png)

2.1）针对分出来的子空间，继续进行特征选择。

幸运的是，我们只有一个子空间<img src="https://latex.codecogs.com/gif.latex?D_2" title="D_2" />需要开发。<img src="https://latex.codecogs.com/gif.latex?D_2" title="D_2" />空间的数据缩小为：

![](/uploads/decision_tree_4.png)

此时<img src="https://latex.codecogs.com/gif.latex?D_2" title="D_2" />空间处于初始状态，即没有选择任何特征，它的信息熵为:

<img src="https://latex.codecogs.com/gif.latex?H(D_2)=-(\frac{3}{9}log_2\frac{3}{9}&plus;\frac{6}{9}log_2\frac{6}{9})=-0.918" />

2.2）假设引入<font size="5">“年龄段”</font>特征。设“年龄段”为<img src="https://latex.codecogs.com/gif.latex?A_1" />,使用上述②式，在<img src="https://latex.codecogs.com/gif.latex?D_2" title="D_2" />数据空间计算条件熵：

P(青年，放贷)=<img src="https://latex.codecogs.com/gif.latex?\frac{1}{9}"  />&emsp;&emsp;&emsp;P(放贷|青年)=<img src="https://latex.codecogs.com/gif.latex?\frac{1}{4}"  />&emsp;&emsp;&emsp;P(青年，不放贷)=<img src="https://latex.codecogs.com/gif.latex?\frac{3}{9}"  />&emsp;&emsp;&emsp;P(不放贷|青年)=<img src="https://latex.codecogs.com/gif.latex?\frac{3}{4}"  />

P(中年，放贷)=<img src="https://latex.codecogs.com/gif.latex?\frac{0}{9}"  />&emsp;&emsp;&emsp;P(放贷|中年)=<img src="https://latex.codecogs.com/gif.latex?\frac{2}{2}"  />&emsp;&emsp;&emsp;P(中年，不放贷)=<img src="https://latex.codecogs.com/gif.latex?\frac{2}{9}"  />&emsp;&emsp;&emsp;P(不放贷|中年)=<img src="https://latex.codecogs.com/gif.latex?\frac{2}{2}"  />

P(老年，放贷)=<img src="https://latex.codecogs.com/gif.latex?\frac{2}{9}"  />&emsp;&emsp;&emsp;P(放贷|老年)=<img src="https://latex.codecogs.com/gif.latex?\frac{2}{3}"  />&emsp;&emsp;&emsp;P(老年，不放贷)=<img src="https://latex.codecogs.com/gif.latex?\frac{1}{9}"  />&emsp;&emsp;&emsp;P(不放贷|老年)=<img src="https://latex.codecogs.com/gif.latex?\frac{1}{3}"  />

<img src="https://latex.codecogs.com/gif.latex?H(D_2|A_1)=-(\frac{1}{9}log_2\frac{1}{4}&plus;\frac{3}{9}log_2\frac{3}{4})-(\frac{0}{9}log_2\frac{2}{2}&plus;\frac{2}{9}log_2\frac{2}{2})-(\frac{2}{9}log_2\frac{2}{3}&plus;\frac{1}{9}log_2\frac{1}{3})=0.667" />

=>

<img src="https://latex.codecogs.com/gif.latex?g(D_2,A_1)=H(D_2)-H(D_2|A_1)=0.918-0.667=0.251"  />

2.3）假设引入<font size="5">“有工作与否”</font>特征。设“有工作与否”为<img src="https://latex.codecogs.com/gif.latex?A_2" />,使用上述②式，计算条件熵：

P(有工作，放贷)=<img src="https://latex.codecogs.com/gif.latex?\frac{3}{9}"  />&emsp;&emsp;&emsp;P(放贷|有工作)=<img src="https://latex.codecogs.com/gif.latex?\frac{3}{3}"  />&emsp;&emsp;&emsp;P(有工作，不放贷)=<img src="https://latex.codecogs.com/gif.latex?\frac{0}{9}"  />&emsp;&emsp;&emsp;P(不放贷|有工作)=<img src="https://latex.codecogs.com/gif.latex?\frac{0}{3}"  />

P(没工作，放贷)=<img src="https://latex.codecogs.com/gif.latex?\frac{0}{9}"  />&emsp;&emsp;&emsp;P(放贷|没工作)=<img src="https://latex.codecogs.com/gif.latex?\frac{0}{6}"  />&emsp;&emsp;&emsp;P(没工作，不放贷)=<img src="https://latex.codecogs.com/gif.latex?\frac{6}{9}"  />&emsp;&emsp;&emsp;P(不放贷|没工作)=<img src="https://latex.codecogs.com/gif.latex?\frac{6}{6}"  />

<img src="https://latex.codecogs.com/gif.latex?H(D_2|A_2)=-(\frac{3}{9}log_2\frac{3}{3}&plus;\frac{0}{9}log_2\frac{0}{3})-(\frac{0}{9}log_2\frac{0}{6}&plus;\frac{6}{9}log_2\frac{6}{6})=0" />

=>

<img src="https://latex.codecogs.com/gif.latex?g(D_2,A_2)=H(D_2)-H(D_2|A_2)=0.918-0=0.918"  />

2.4）假设引入<font size="5">“信贷情况”</font>特征。设“信贷情况”为<img src="https://latex.codecogs.com/gif.latex?A_3" />,使用上述②式，计算条件熵：

P(信贷一般，放贷)=<img src="https://latex.codecogs.com/gif.latex?\frac{0}{9}"  />&emsp;&emsp;&emsp;P(放贷|信贷一般)=<img src="https://latex.codecogs.com/gif.latex?\frac{0}{4}"  />&emsp;&emsp;&emsp;P(信贷一般，不放贷)=<img src="https://latex.codecogs.com/gif.latex?\frac{4}{9}"  />&emsp;&emsp;&emsp;P(不放贷|信贷一般)=<img src="https://latex.codecogs.com/gif.latex?\frac{4}{4}"  />

P(信贷好，放贷)=<img src="https://latex.codecogs.com/gif.latex?\frac{2}{9}"  />&emsp;&emsp;&emsp;P(放贷|信贷好)=<img src="https://latex.codecogs.com/gif.latex?\frac{2}{4}"  />&emsp;&emsp;&emsp;P(信贷好，不放贷)=<img src="https://latex.codecogs.com/gif.latex?\frac{2}{9}"  />&emsp;&emsp;&emsp;P(不放贷|信贷好)=<img src="https://latex.codecogs.com/gif.latex?\frac{2}{4}"  />

P(信贷非常好，放贷)=<img src="https://latex.codecogs.com/gif.latex?\frac{1}{9}"  />&emsp;&emsp;&emsp;P(放贷|信贷非常好)=<img src="https://latex.codecogs.com/gif.latex?\frac{1}{1}"  />&emsp;&emsp;&emsp;P(信贷非常好，不放贷)=<img src="https://latex.codecogs.com/gif.latex?\frac{0}{9}"  />&emsp;&emsp;&emsp;P(不放贷|信贷非常好)=<img src="https://latex.codecogs.com/gif.latex?\frac{0}{1}"  />

<img src="https://latex.codecogs.com/gif.latex?H(D_2|A_3)=-(\frac{0}{9}log_2\frac{0}{4}&plus;\frac{4}{9}log_2\frac{4}{4})-(\frac{2}{9}log_2\frac{2}{4}&plus;\frac{2}{9}log_2\frac{2}{4})-(\frac{1}{9}log_2\frac{1}{1}&plus;\frac{0}{9}log_2\frac{0}{1})=0.444" />

=>

<img src="https://latex.codecogs.com/gif.latex?g(D_2,A_3)=H(D_2)-H(D_2|A_3)=0.918-0.444=0.474" />

2.5）比较<img src="https://latex.codecogs.com/gif.latex?g(D_2,A_1)" />，<img src="https://latex.codecogs.com/gif.latex?g(D_2,A_2)" />，<img src="https://latex.codecogs.com/gif.latex?g(D_2,A_3)" />&emsp;=>&emsp;<img src="https://latex.codecogs.com/gif.latex?g(D_2,A_2)>g(D_2,A_3)>g(D_2,A_1)"  />

所以我们选择的特征是“有工作与否”，并且注意，该特征的信息增益是<img src="https://latex.codecogs.com/gif.latex?g(D_2,A_2)=0.918"  />，意味着，全部信息被衰减，流程图应该停止了。我们绘图看看是否如此：

![](/uploads/decision_tree_5.png)

的确如此！

我们比较此图和初始的流程图，可知，采用信息增益，可以尽可能等减少分支和叶子，迅速决策。

## 决策树生成算法

常见的决策树生成算法有ID3算法，C4.5算法和CART算法

### ID3算法
ID3算法使用的是信息增益，其运算过程正是上节所描述的例子！

### C4.5算法
C4.5与ID3的区别仅仅是使用<font size=4>信息增益比</font> 而非信息增益。为什么要用信息增益比呢？

因为在使用信息增益，会导致特征选择时偏向于选择取值较多的那个特征，而那个特征可能是没有意义的。比如编号，身份证号，星期几。如下的用星期作为特征，可以产生5个分支，使用ID3算法的信息增益可以将所有信息熵衰减掉，直接生成五个叶子：

![](/uploads/decision_tree_6.png)

![](/uploads/decision_tree_6.png)

这是非常严重的过拟合。为了避免这种情况的产生，我们必须增加与分支数（选定特征的取值数）相关的惩罚项，形成信息增益比：

<img src="https://latex.codecogs.com/gif.latex?g_R(D,A)=\frac{g(D,A)}{H_A(D)}" />

其中惩罚项是<img src="https://latex.codecogs.com/gif.latex?H_A(D)=-\sum_{i=1}^{n}\frac{|D_i|}{|D|}log_2\frac{|D_i|}{|D|}" title="H_A(D)=-\sum_{i=1}^{n}\frac{|D_i|}{|D|}log_2\frac{|D_i|}{|D|}" />，<img src="https://latex.codecogs.com/gif.latex?n"  />是所选特征的取值数，<img src="https://latex.codecogs.com/gif.latex?D_i"  />是子空间，此例中<img src="https://latex.codecogs.com/gif.latex?\frac{|D_i|}{|D|}=\frac{1}{5}" title="\frac{|D_i|}{|D|}=\frac{1}{5}" />

### CART算法

将会在之后的文章中单独讲解

### 说明
当然，需要<font size=5>注意</font>的是：无论使用哪种算法，对于大数据来说，我们的决策树的层数可能非常多，如果我们分解到没有特征可分的地步会过拟合，且时间消耗非常大。 通常的做法是设置一个信息增益阈值<img src="https://latex.codecogs.com/gif.latex?\varepsilon" title="\varepsilon" />，当某个特征引入后信息增益小于该阈值<img src="https://latex.codecogs.com/gif.latex?\varepsilon" title="\varepsilon" />，我们就可以停止树分裂，并把原始数据集中<font size=4>数量最多</font>的那个类别作为叶子节点


## 附录：联合概率分布和条件概率分布的区别
考虑两个变量：天晴与否，花开与否

它们组合成四种事件：
1）天晴了A1，花开了B1
2）天晴了A1，花没开B2
3）天没晴A2，花开了B1
4）天没晴A2，花没开B2

联合概率分别描述这四种情况:

<img src="https://latex.codecogs.com/gif.latex?P(A1,B1)=\frac{1}{4}"  />，&emsp;<img src="https://latex.codecogs.com/gif.latex?P(A1,B2)=\frac{1}{4}"  />，&emsp;<img src="https://latex.codecogs.com/gif.latex?P(A2,B1)=\frac{1}{4}"  />，&emsp;<img src="https://latex.codecogs.com/gif.latex?P(A2,B2)=\frac{1}{4}"  />，

条件概率分布描述的是某个事件已经发生了，比如B1已经发生了，就扼杀了2)，4)的可能，我们就只能从1），3）两种情况去考虑：

<img src="https://latex.codecogs.com/gif.latex?P(A1|B1)=\frac{1}{2}" />，&emsp;<img src="https://latex.codecogs.com/gif.latex?P(A2|B1)=\frac{1}{2}" />

条件概率和联合概率的数学关系：

<img src="https://latex.codecogs.com/gif.latex?P(A=A_i|B=B_i)=\frac{P(A=A_i,B=B_i)}{P(B=B_i)}"  />


## 参考
《统计方法学》by 李航







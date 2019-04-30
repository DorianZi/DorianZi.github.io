---
title: CRF 条件随机场
date: 2019-04-10 20:18:41
tags: ["Algorithm","NLP"]
categories: Technic
---

# CRF的定义

CRF指的是Conditional Random Field,条件随机场。可以用HMM来帮助理解。

![](/uploads/CRF_1.png)

直观看图和公式，可以知道HMM里面观测序列的每一项都只跟发射它的状态序列项有关，即<img src="https://latex.codecogs.com/gif.latex?y_1"  />对<img src="https://latex.codecogs.com/gif.latex?y_1" title="x_1" />，<img src="https://latex.codecogs.com/gif.latex?y_2"  />对<img src="https://latex.codecogs.com/gif.latex?x_2"  />，<img src="https://latex.codecogs.com/gif.latex?x_1"  />对<img src="https://latex.codecogs.com/gif.latex?x_3"  />。而CRF里面观测序列的每一项都可以和任何一个状态序列项有关(因为无向)，即<img src="https://latex.codecogs.com/gif.latex?y_i" title="y_i" />对<img src="https://latex.codecogs.com/gif.latex?x_1,x_2,x_3" title="x_1,x_2,x_3" />。

另外的区别是：HMM是有向图，严格定义了y的有序性 CRF是无向图，y无序（图中是线性链式，为CRF的特殊情况）。HMM是生成模型，通过求联合概率获得；CRF是判别模型，通过条件概率求得。在如词性标注上的应用中CRF更合理，因为它直接求某个标注的概率，而HMM需要先算联合概率再转而求目标概率。

CRF的标准定义是：

形式化定义：设<img src="https://latex.codecogs.com/gif.latex?X=(x_1,x_2,...,x_n),Y=(y_1,y_2,...,y_n)" title="X=(x_1,x_2,...,x_n),Y=(y_1,y_2,...,y_n)" />均为线性链表示的随机变量序列，若给在定随机变量序列X的条件下，随机变量序列Y的条件概率分布P(Y|X)构成条件随机场，即满足马尔科夫性：<img src="https://latex.codecogs.com/gif.latex?P(y_i|x,y_1,...,y_{i-1},y_{i&plus;1},...,y_n)=P(y_i|x,y_{i-1},y_{i&plus;1}),\&space;\&space;i=1,2,...,n" />，则称P(Y|X)是线性链条件随机场。

注：本文只谈线性链CRF，即上图中所示的CRF结构

# CRF在词性标注的应用

我们通过CRF在词性标注上的应用来理解CRF算法原理。

有这样一个句子： Dorian is a good boy 正确的词性标注为：名词-动词-名词。现在我们通过CRF来实现它：

![](/uploads/CRF_2.png)

## 特征函数评分
首先需要建立两种特征函数，转移特征函数和状态特征函数，每种特征函数有很多个。转移特征函数用来给相邻（只用相邻的就够了，这也是线性链CRF的特点）词性的组合打分，状态特征函数用来给当前位置的特征打分。然后给每个特征函数的打分一个权重，最后相加起来得到综合分数：

<img src="https://latex.codecogs.com/gif.latex?\sum_{i,k}^{&space;}\lambda_kt_k(y_{i-1},y_{i},x,i)&plus;\sum_{i,l}^{&space;}\mu_ls_l(y_i,x,i)"/>

它展开写应该是：

<img src="https://latex.codecogs.com/gif.latex?\sum_{k=1}^{m}\lambda_k\sum_{i=2}^{n}t_k(y_{i-1},y_{i},x,i)&plus;\sum_{l=1}^{h}\mu_l\sum_{i=2}^{n}s_l(y_i,x,i)" title="\sum_{k=1}^{m}\lambda_k\sum_{i=2}^{n}t_k(y_{i-1},y_{i},x,i)+\sum_{l=1}^{h}\mu_l\sum_{i=2}^{n}s_l(y_i,x,i)" />

其中i从2到n，表示从这句话的第2个词开始到最后一个词，每一段都使用这个转移特征函数进行打分，并相加，得到整个句子的评分，然后乘以一个统一的权重<img src="https://latex.codecogs.com/gif.latex?t_k" title="t_k" />；k从1到m，表示有m个不同的转移特征函数，每个函数都要对这个句子进行一遍打分，进行加权。

特征函数是怎么工作的呢？

**问题：**我们要看看“Dorian is a good boy”这个句子的词性标注为“动词-动词-名词-名词-名词”的话，也就是“v-v-n-n-n”，会被打多少分（显然我们期待它的打分很低）。

假设我们第一个转移特征函数<img src="https://latex.codecogs.com/gif.latex?t_1(y_{i-1},y_{i},x,i)" title="t_1(y_{i-1},y_{i},x,i)" />能够判断词性连接的合理性,它使用到前两个词性上是：

<img src="https://latex.codecogs.com/gif.latex?t_1(v,v,''Dorian\&space;is\&space;a\&space;good\&space;boy'',2)" />

它应该返回0分或负分，比如-1。因为它知道两个动词相邻是不会出现的。

再比如第三个状态特征函数<img src="https://latex.codecogs.com/gif.latex?s_3(y_{i},x,i)"  />能够判断词性的存在性,它使用到第二个词性上是：

<img src="https://latex.codecogs.com/gif.latex?s_3(v,''Dorian\&space;is\&space;a\&space;good\&space;boy'',2)"  />

它应该返回1分，因为它知道“is”是个动词。这一个词分高不代表什么，我们要看的是总分而且是加权后的。

再看一个特别的：

<img src="https://latex.codecogs.com/gif.latex?s_3(n,''Dorian\&space;is\&space;a\&space;good\&space;boy'',4)"  />

这个时候有可能返回0.5分，为什么？因为“good”的确有名词的词性。所以这里仅仅通过状态特征函数是判断不出来的，我们还需要前后语境的辅助，这就是为什么我们需要转移状态特征函数。

好，总之，所有评分结束并加权之后，我们得到了评分：

<img src="https://latex.codecogs.com/gif.latex?score(v-v-n-n-n)=\sum_{k=1}^{m}\lambda_k\sum_{i=2}^{5}t_k(y_{i-1},y_{i},''Dorian\&space;is\&space;a\&space;good\&space;boy'',i)&plus;\sum_{l=1}^{h}\mu_l\sum_{i=2}^{5}s_l(y_i,''Dorian\&space;is\&space;a\&space;good\&space;boy'',i)"  />

接下来，我们需要计算其它很多种情况的评分，比如<img src="https://latex.codecogs.com/gif.latex?score(v-v-v-n-n)"  />，比如<img src="https://latex.codecogs.com/gif.latex?score(v-p-v-art-n)"  />。最后我们要比较所有情况里面，谁的评分最高，那么谁就是最后的词性标注结果。如果特征函数设置正确的话，那么得分最高的会是：

<img src="https://latex.codecogs.com/gif.latex?score(n-v-art-adj-n)"  />，也就是score("名词-动词-冠词-形容词-名词")

## 规范化概率

掌握上面的计算，就能基于给定的模型进行词性标注了。

如果想获得某种标注情况出现的概率，我们可以将评分归一化，或者规范化，即考虑这种标注序列在所有标注序列里面的比重，也就是出现概率。选择先指数化，然后除以所有标注情况评分指数化的总和：

<img src="https://latex.codecogs.com/gif.latex?P(y|x)=\frac{1}{Z(x)}&space;exp&space;\left(&space;\sum_{i,k}^{&space;}\lambda_kt_k(y_{i-1},y_{i},x,i)&plus;\sum_{i,l}^{&space;}\mu_ls_l(y_i,x,i)&space;\right&space;)"  />

其中<img src="https://latex.codecogs.com/gif.latex?Z(x)"  />为所有情况的指数化总和（注意，是先指数化再求和）：

<img src="https://latex.codecogs.com/gif.latex?Z(x)=\sum_{y}^{&space;}&space;exp&space;\left(&space;\sum_{i,k}^{&space;}\lambda_kt_k(y_{i-1},y_{i},x,i)&plus;\sum_{i,l}^{&space;}\mu_ls_l(y_i,x,i)&space;\right&space;)" />

## 线性统一化
如果我们统一一下两种特征函数（假设两种函数分别有<img src="https://latex.codecogs.com/gif.latex?K_1,K_2" title="K_1,K_2" />个，为便于后面计算，再设<img src="https://latex.codecogs.com/gif.latex?K=K_1+K_2" />）：

创造分段函数，统一两种特征函数：

<img src="https://latex.codecogs.com/gif.latex?f_k(y_{i-1},y_{i},x,i)=\left\{\begin{matrix}&space;\&space;t_k(y_{i-1},y_i,x,i),\&space;\&space;k=1,2,...,K_1\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\\&space;\&space;s_l(y_i,x,i),\&space;\&space;k=K_1&plus;l;\&space;l=1,2,...,K_2&space;\end{matrix}\right."  />

那么它们在一个标注里面求和的表达式，可以被统一为：

<img src="https://latex.codecogs.com/gif.latex?f_k(y,x)=\sum_{i=1}^{n}f_k(y_{i-1},y_i,x,i),\&space;\&space;k=1,2,...,K" title="f_k(y,x)=\sum_{i=1}^{n}f_k(y_{i-1},y_i,x,i),\ \ k=1,2,...,K" />

接下来需要乘以权重，我们把权重也给统一为分段函数：

<img src="https://latex.codecogs.com/gif.latex?w_k=\left\{\begin{matrix}&space;\&space;\lambda_k,\&space;\&space;k=1,2,...,K_1\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\\&space;\&space;\mu_l,\&space;\&space;k=K_1&plus;l;\&space;l=1,2,...,K_2&space;\end{matrix}\right."  />

那么我们规范化的条件随机场可以统一为：

<img src="https://latex.codecogs.com/gif.latex?P(y|x)=\frac{1}{Z(x)}exp\sum_{k=1}^{K}w_kf_k(y,x)"  />

其中 <img src="https://latex.codecogs.com/gif.latex?Z(x)=\sum_{y}^{&space;}exp\sum_{k=1}^{K}w_kf_k(y,x)"  />

可以看到它的加权过程变成了线性的形式。而对于线性表达，我们很容易想到用向量内积来表示，所以不妨令：

<img src="https://latex.codecogs.com/gif.latex?w=(w_1,w_2,...,w_{K})" title="w=(w_1,w_2,...,w_{K})" />

<img src="https://latex.codecogs.com/gif.latex?F(y,x)=(f_1(y,x),f_1(y,x),...,f_K(y,x))T" title="F(y,x)=(f_1(y,x),f_1(y,x),...,f_K(y,x))T" />

那么我们规范化的条件随机场可以进一步统一为：

<img src="https://latex.codecogs.com/gif.latex?P(y|x)=\frac{1}{Z_w(x)}exp(wF(y,x))"  />

其中 <img src="https://latex.codecogs.com/gif.latex?Z_w(x)=\sum_{y}^{&space;}exp(wF(y,x))"  />

# CRF的模型训练

上节讲的是在给定了的模型下，如何使用它进行标注。这一节讲，如何通过训练数据集，训练CRF标注模型。
所谓的训练，指的是训练权重，也就是我们已经有特征函数的情况（对同一种语言和语境，特征函数可以固定）下，训练权重<img src="https://latex.codecogs.com/gif.latex?w=(w_1,w_2,...,w_K)" title="w=(w_1,w_2,...,w_K)" />

基于<font size="4">极大似然</font>思想，通俗地说就是，我们要找到一组<img src="https://latex.codecogs.com/gif.latex?w" title="w" />，使得训练集里面任何一组数据（如x="Dorian plays football",y="名词-动词-名词"就是一组数据）的概率<img src="https://latex.codecogs.com/gif.latex?P(y|x)" title="P(y|x)" />尽可能大，这样才说明该模型能够很“自信”地进行标注。为此，我们只要使得所有样本概率的乘积<img src="https://latex.codecogs.com/gif.latex?\prod_{x,y}^{&space;}P(y|x)" title="\prod_{x,y}^{ }P(y|x)" />最大化即可，取对数就是<img src="https://latex.codecogs.com/gif.latex?log\prod_{x,y}^{&space;}P(y|x)"  />。其中<img src="https://latex.codecogs.com/gif.latex?\prod_{x,y}^{&space;}"  />表示对样本数据的迭代。

使用最大熵模型，引入训练数据的经验概率分布<img src="https://latex.codecogs.com/gif.latex?\tilde{P}(x,y)" title="\tilde{P}(x,y)" />，得到<img src="https://latex.codecogs.com/gif.latex?log\prod_{x,y}^{&space;}P(y|x)^{\tilde{P}(x,y)}"  />，该对数似然函数就是最大熵模型的极大似然估计：

<img src="https://latex.codecogs.com/gif.latex?L(w)=L_{\tilde{P}}(P_w)=log\prod_{x,y}^{&space;}P(y|x)^{\tilde{P}(x,y)}=\sum_{x,y}^{&space;}\tilde{P}(x,y)logP(y|x)" title="L(w)=L_{\tilde{P}}(P_w)=log\prod_{x,y}^{ }P(y|x)^{\tilde{P}(x,y)}=\sum_{x,y}^{ }\tilde{P}(x,y)logP(y|x)" />

接下来采用改进的迭代尺度法（即IIS法），可以求解该最大熵模型的参数迭代式，这里就不介绍了。

以上。










---
title: Maximum Entropy Model 最大熵模型
date: 2019-04-12 08:34:34
tags: ["Algorithm"]
categories: Technic
---

最大熵模型的思想是：数据集没有约束（或满足了已知约束之后）的情况下，认为数据的概率分布是均匀的，没有偏向说哪些数据是概率更大的，这种情况也代表整个数据集是熵最大的。这是在缺乏信息的情况下能够做到的最合理的“认为”，此时通过求熵最大，来求得模型参数。这样求得的模型在进行预测时，尽管精度不保证更高，但它能覆盖到更多的情况，因为它在训练时候没有“偏见”。

举个例子，一个六面色子，在没有任何信息的情况下，假定所有数字被掷到的概率相等，为<img src="https://latex.codecogs.com/gif.latex?\frac{1}{6}" title="\frac{1}{6}" />，这个假设是最合理的，也是使得熵最大的。

# 最大熵模型

设有样本集<img src="https://latex.codecogs.com/png.latex?\{(x^{(1)},y^{(1)}),(x^{(2)},y^{(2)}),...,(x^{(N)},y^{(N)})\}" title="\{(x^{(1)},y^{(1)}),(x^{(2)},y^{(2)}),...,(x^{(N)},y^{(N)})\}" />，其中类别是<img src="https://latex.codecogs.com/png.latex?\(y^{(1)},y^{(2)}),...,y^{(N)})" />（不一定是N个类别，里面可以有相等值），需要求解合理分类器。

选择使用最大熵模型，也就是选择了软分类器。软分类器的意思是通过求解不同类别的概率<img src="https://latex.codecogs.com/png.latex?P(Y=y^{(j)}|X=x^{(i)}),\&space;\&space;\&space;j=1,2,...,m" title="P(Y=y^{(j)}|X=x^{(i)}),\ \ \ j=1,2,...,m" />并从中挑选出概率最大的那个，来决定最终分类的类别。所以这里我们的目标是写出<img src="https://latex.codecogs.com/png.latex?P(y|x)" title="P(Y|X)" />函数的表达式。而最大熵的思想是：通过最大化训练集的条件熵，可以获得最优的<img src="https://latex.codecogs.com/png.latex?P(y|x)" title="P(Y|X)" />表达式——

<img src="https://latex.codecogs.com/png.latex?\underset{P&space;\epsilon&space;C&space;}{max}\&space;H(P)=-\sum_{x,y}\tilde{P}(x)P(y|x)logP(y|x)" title="\underset{P \epsilon C }{max}\ H(P)=-\sum_{x,y}\tilde{P}(x)P(y|x)logP(y|x)" />

也就是找到合适的P也就是<img src="https://latex.codecogs.com/png.latex?P(y|x)" title="P(y|x)" />，使得上面的条件熵最大。其中<img src="https://latex.codecogs.com/png.latex?\tilde{P}(x)" title="\tilde{P}(X)" />为来自数据样本的经验分布，计算方法是：<img src="https://latex.codecogs.com/png.latex?\tilde{P}(X=x)=\frac{v(X=x)}{N}" title="\tilde{P}(X=x)=\frac{v(X=x)}{N}" />即在样本集中，x出现的频次在样本总数N里的占比。因为我们没法得到<img src="https://latex.codecogs.com/png.latex?P(x)" title="P(X)" />，所以用经验分布代替。

**注：**这里的<img src="https://latex.codecogs.com/png.latex?\sum_{x,y}^{&space;}" title="\sum_{x,y}^{ }" />表示遍历样本集的去重后的数据，而非遍历变量取值范围。比如变量取值范围是<img src="https://latex.codecogs.com/png.latex?X=1,2,3,4,5\&space;\&space;Y=1,2,3,4" title="X=1,2,3,4,5\ \ Y=1,2,3,4" />而样本集为<img src="https://latex.codecogs.com/png.latex?\{(1,2),(1,2),(3,4),(5,4)\}" title="\{(1,2),(1,2),(3,4),(5,4)\}" />，则<img src="https://latex.codecogs.com/png.latex?\sum_{x,y}^{&space;}" title="\sum_{x,y}^{ }" />表示<img src="https://latex.codecogs.com/png.latex?\{(1,2),(3,4),(5,4)\}" title="\{(1,2),(3,4),(5,4)\}" />的遍历

到目前为止，只有<img src="https://latex.codecogs.com/png.latex?\underset{P&space;\epsilon&space;C&space;}{max}\&space;H(P) " />还不足以计算出P，我们还应该从数据中探索约束条件。

## 特征函数及约束条件

首先第一点，样本集里的数据一定是符合某种数据规则的，也就是<img src="https://latex.codecogs.com/png.latex?x^{(i)}" title="x^{(i)}" />和<img src="https://latex.codecogs.com/png.latex?y^{(i)}"  />之间是有某种关系的，不然任何组合都可以成为样本数据，那么求模型就没有任何意义了。我们把<img src="https://latex.codecogs.com/png.latex?x^{(i)}" title="x^{(i)}" />和<img src="https://latex.codecogs.com/png.latex?y^{(i)}"  />之间的关系定义为特征函数：

<img src="https://latex.codecogs.com/png.latex?f(x,y)=\left\{\begin{matrix}&space;1,\&space;\&space;if\&space;x\&space;has\&space;relationship\&space;with\&space;y\\&space;0,\&space;\&space;otherwise\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\end{matrix}\right." title="f(x,y)=\left\{\begin{matrix} 1,\ \ if\ x\ has\ relationship\ with\ y\\ 0,\ \ otherwise\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \end{matrix}\right." />

我们可能需要很多个不同特征的组合，因为x和y的关系可能不止一种。举个例子，基于下面的数据集判断“resume”的在句子里的意思是“简历”（设为0）还是“继续”（设为1）：

<img src="https://latex.codecogs.com/png.latex?\begin{align*}&space;&(x^{(1)},y^{(1)})=(''They\&space;will\&space;resume\&space;negotiations\&space;today'',0)\\&space;&(x^{(2)},y^{(2)})=(''He\&space;resumes\&space;study'',0)\\&space;&(x^{(3)},y^{(3)})=(''The\&space;resume\&space;looks\&space;good'',1)&space;\end{align*}" />

定义2个它的函数：

1）当resume后面是名词，则resume为“继续”：

<img src="https://latex.codecogs.com/png.latex?f_1(x,y)=\left\{\begin{matrix}&space;1,\&space;\&space;if\&space;noun\&space;next\&space;to\&space;it\\&space;0,\&space;\&space;otherwise\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\end{matrix}\right." title="f(x,y)=\left\{\begin{matrix} 1,\ \ if\ noun\ next\ to\ it\\ 0,\ \ otherwise\ \ \ \ \ \ \ \ \ \ \ \end{matrix}\right." />

2）当resume后面是动词，则resume为“简历”:

<img src="https://latex.codecogs.com/png.latex?f_2(x,y)=\left\{\begin{matrix}&space;1,\&space;\&space;if\&space;verb\&space;next\&space;to\&space;it\\&space;0,\&space;\&space;otherwise\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\end{matrix}\right." title="f(x,y)=\left\{\begin{matrix} 1,\ \ if\ verb\ next\ to\ it\\ 0,\ \ otherwise\ \ \ \ \ \ \ \ \ \ \end{matrix}\right." />

则特征函数可以表示数据的特性：
<img src="https://latex.codecogs.com/png.latex?\begin{align*}&space;&f_1(x^{(1)},y^{(1)})=f_1(x^{(2)},y^{(2)})=1\&space;\&space;\&space;\&space;f_1(x^{(3)},y^{(3)})=0\\&space;&f_2(x^{(1)},y^{(1)})=f_2(x^{(2)},y^{(2)})=0\&space;\&space;\&space;\&space;f_2(x^{(3)},y^{(3)})=1&space;\end{align*}" title="\begin{align*} &f_1(x^{(1)},y^{(1)})=f_1(x^{(2)},y^{(2)})=1\ \ \ \ f_1(x^{(3)},y^{(3)})=0\\ &f_2(x^{(1)},y^{(1)})=f_2(x^{(2)},y^{(2)})=0\ \ \ \ f_2(x^{(3)},y^{(3)})=1 \end{align*}" />

特征函数<img src="https://latex.codecogs.com/png.latex?f(x,y)" title="f(x,y)" />在数据集上关于经验分布<img src="https://latex.codecogs.com/png.latex?\tilde{P}(x,y)" title="\tilde{P}(x,y)" />的期望为：

<img src="https://private.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20E%7B_%7B%5Cwidetilde%7Bp%7D%7D%7D%28f%29%3D%5Csum_%7Bx%2Cy%7D%5Cwidetilde%7BP%7D%28x%2Cy%29f%28x%2Cy%29">

特征函数<img src="https://latex.codecogs.com/png.latex?f(x,y)" title="f(x,y)" />在数据集上关于分布<img src="https://latex.codecogs.com/png.latex?P(x,y)" title="\tilde{P}(x,y)" />的期望为：

<img src="https://latex.codecogs.com/png.latex?E_{P}(f)=\sum_{x,y}^{&space;}P(x,y)f(x,y)" title="E_{P}(f)=\sum_{x,y}^{ }P(x,y)f(x,y)" />

由于我们得不到<img src="https://latex.codecogs.com/png.latex?P(x,y)" title="P(x,y)" />，所以只好通过<img src="https://latex.codecogs.com/png.latex?\tilde{P}(x)" title="\tilde{P}(x)" />替换：<img src="https://latex.codecogs.com/png.latex?P(x,y)=P(x)P(y|x)\approx&space;\tilde{P}(x)P(y|x)" title="P(x,y)=P(x)P(y|x)\approx \tilde{P}(x)P(y|x)" />，所以有：

<img src="https://latex.codecogs.com/png.latex?E_{P}(f)=\sum_{x,y}^{&space;}\tilde{P}(x)P(y|x)f(x,y)"  />

如果我们通过训练数据能训练出模型，那么两个特征函数期望相等：

<img src="https://latex.codecogs.com/png.latex?\sum_{x,y}^{&space;}\tilde{P}(x,y)f(x,y)=\sum_{x,y}^{&space;}\tilde{P}(x)P(y|x)f(x,y)"  />

于是我们得到了第一个<font size="4">约束条件</font>。

此外容易得到另一个约束条件，对所有可能，概率和为1：

<img src="https://latex.codecogs.com/png.latex?\sum_{y}^{&space;}P(y|x)=1" title="\sum_{y}^{ }P(y|x)=1" />

于是我们得到了第二个<font size="4">约束条件</font>。

## 求解法（一）：拉格朗日对偶化
综上，我们的问题表述为：

<img src="https://latex.codecogs.com/png.latex?\begin{align*}&space;&\underset{P\epsilon&space;C}{max}\&space;H(P)=-\sum_{x,y}\tilde{P}(x)P(y|x)logP(y|x)\\&space;&s.t.\&space;E_p(f_i)=E_{\tilde{P}}(f_i),\&space;i=1,2,...,n\\&space;&\&space;\&space;\&space;\&space;\&space;\sum_{y}P(y|x)=1&space;\end{align*}" title="\begin{align*} &\underset{P\epsilon C}{max}\ H(P)=-\sum_{x,y}\tilde{P}(x)P(y|x)logP(y|x)\\ &s.t.\ E_p(f_i)=E_{\tilde{P}}(f_i),\ i=1,2,...,n\\ &\ \ \ \ \ \sum_{y}P(y|x)=1 \end{align*}" />

将最大问题改写为最小问题：

<img src="https://latex.codecogs.com/png.latex?\begin{align*}&space;&\underset{P\epsilon&space;C}{min}\&space;-H(P)=-\sum_{x,y}\tilde{P}(x)P(y|x)logP(y|x)\\&space;&s.t.\&space;E_p(f_i)=E_{\tilde{P}}(f_i),\&space;i=1,2,...,n\\&space;&\&space;\&space;\&space;\&space;\&space;\sum_{y}P(y|x)=1&space;\end{align*}" title="\begin{align*} &\underset{P\epsilon C}{min}\ -H(P)=-\sum_{x,y}\tilde{P}(x)P(y|x)logP(y|x)\\ &s.t.\ E_p(f_i)=E_{\tilde{P}}(f_i),\ i=1,2,...,n\\ &\ \ \ \ \ \sum_{y}P(y|x)=1 \end{align*}" />

使用拉格朗日法，引进乘子<img src="https://latex.codecogs.com/png.latex?w_0,w_1,w_2,...w_n" title="w_0,w_1,w_2,...w_n" />，得到拉格朗日函数：

<img src="https://latex.codecogs.com/png.latex?\begin{align*}&space;L(P,w)=&-H(P)&plus;w_0\left&space;(&space;1-\sum_{y}P(y|x)&space;\right&space;)&plus;\sum_{i=1}^{n}w_i(E_{\tilde{P}}(f_i)-E_P(f_i))\\&space;=&\sum_{x,y}\tilde{P}(x)P(y|x)logP(y|x)&plus;w_0\left&space;(&space;1-\sum_{y}P(y|x)&space;\right&space;)&plus;\sum_{i=1}^{n}w_i\left&space;(&space;\sum_{x,y}\tilde{P}(x,y)f_i(x,y)-\sum_{x,y}\tilde{P}(x)P(y|x)f_i(x|y)&space;\right&space;)&space;\end{align*}" title="\begin{align*} L(P,w)=&-H(P)+w_0\left ( 1-\sum_{y}P(y|x) \right )+\sum_{i=1}^{n}w_i(E_{\tilde{P}}(f_i)-E_P(f_i))\\ =&\sum_{x,y}\tilde{P}(x)P(y|x)logP(y|x)+w_0\left ( 1-\sum_{y}P(y|x) \right )+\sum_{i=1}^{n}w_i\left ( \sum_{x,y}\tilde{P}(x,y)f_i(x,y)-\sum_{x,y}\tilde{P}(x)P(y|x)f_i(x|y) \right ) \end{align*}" />

问题被转换为：

<img src="https://latex.codecogs.com/png.latex?\underset{P\epsilon&space;C}{min}\&space;\underset{w}{max}\&space;L(P,w)" title="\underset{P\epsilon C}{min}\ \underset{w}{max}\ L(P,w)" />

因为上面为P的凸函数，则可以进一步转化为对偶问题：

<img src="https://latex.codecogs.com/png.latex?\underset{w}{max}\&space;\underset{\tilde{P\epsilon&space;C}}{min}\&space;L(P,w)" title="\underset{w}{max}\ \underset{\tilde{P\epsilon C}}{min}\ L(P,w)" />&emsp;&emsp;①

先求内部的min,通过对P的偏导数为0求得：

<img src="https://latex.codecogs.com/png.latex?P(y|x)=\frac{1}{Z(x)}exp\left&space;(&space;\sum_{i=1}^{n}w_if_i(x,y)&space;\right&space;)" title="P_w(y|x)=\frac{1}{Z_w(x)}exp\left ( \sum_{i=1}^{n}w_if_i(x,y) \right )" />

其中<img src="https://latex.codecogs.com/png.latex?Z(x)=\sum_{y}exp\left&space;(&space;\sum_{i=1}^{n}w_if_i(x,y)\right&space;)" title="Z(x)=\sum_{y}exp\left ( \sum_{i=1}^{n}w_if_i(x,y)\right )" />

上面的式子中，<img src="https://latex.codecogs.com/png.latex?w_i" title="w_i" />还是未知数，接下来继续对外部的max求解，可以得到最优的<img src="https://latex.codecogs.com/png.latex?w_i" title="w_i" />解，再代入上式，就可以得到<img src="https://latex.codecogs.com/png.latex?P(y|x)" title="P(y|x)" />的表达式了。


## 求解法（二）：极大对数似然估计

除了上面的方法，还有一种更简单的方法可以转换最大熵问题，即转换为对数极大似然估计。
条件概率的极大似然函数，就是希望样本中各个概率乘积最大：

<img src="https://latex.codecogs.com/png.latex?\prod_{i=1}^{N}P(y^{(i)}|x^{(i)})"  />

N个样本中可能有重复的，合并重复样本，一共有n个不重复的值，则上式可以写作：

<img src="https://latex.codecogs.com/png.latex?\prod_{i=1}^{n}P(y^{(i)}|x^{(i)})^{C(x=x^{(i)},y=y^{(i)})}" title="\prod_{i=1}^{n}P(y^{(i)}|x^{(i)})^{C(x=x^{(i)},y=y^{(i)})}" />

其中C为每个值在数据集里出现的次数。

我们只要最大化上式，就可以求得<img src="https://latex.codecogs.com/png.latex?P(y|x)" title="P(y|x)" />的表达式。

不过此处还可以再转换一次：最大化上式跟最大化它的开N次根，是等效的，所以我们将它开N次根：

<img src="https://latex.codecogs.com/png.latex?\prod_{i=1}^{n}P(y^{(i)}|x^{(i)})^{&space;\frac{C(x=x^{(i)},y=y^{(i)})}{N}}&space;=\prod_{i=1}^{n}P(y^{(i)}|x^{(i)})^{\tilde{P}(x,y)}" title="\prod_{i=1}^{n}P(y^{(i)}|x^{(i)})^{ \frac{C(x=x^{(i)},y=y^{(i)})}{N}} =\prod_{i=1}^{n}P(y^{(i)}|x^{(i)})^{\tilde{P}(x,y)}" />

为了便于计算，再对它取log，就得到<font size="4">对数似然函数</font>:

<img src="https://latex.codecogs.com/png.latex?\begin{align*}&space;L_{\tilde{P}}(P_w)=&\sum_{x,y}\tilde{P}(x,y)logP(y|x)\\&space;=&\sum_{x,y}\tilde{P}(x,y)\sum_{i=1}^{n}w_if_i(x,y)-\sum_{x,y}\tilde{P}(x,y)logZ_w(x)\\&space;=&\sum_{x,y}\tilde{P}(x,y)\sum_{i=1}^{n}w_if_i(x,y)-\sum_{x}\tilde{P}(x)logZ_w(x)&space;\end{align*}" />

最大化上式，即<img src="https://latex.codecogs.com/png.latex?\underset{w}{max}\&space;L_{\tilde{P}}(P_w)"  />就是最大熵模型的最优解。为什么？证明如下：

已知上面拉格朗日对偶化问题①为<img src="https://latex.codecogs.com/png.latex?\underset{w}{max}\&space;\underset{\tilde{P\epsilon&space;C}}{min}\&space;L(P,w)"  />，我们只要证明<img src="https://latex.codecogs.com/png.latex?\underset{\tilde{P\epsilon&space;C}}{min}\&space;L(P,w)"  />等于这里的对数似然函数<img src="https://latex.codecogs.com/png.latex?L_{\tilde{P}}(P_w)"  />就好了。

通过代数计算（过程略）可以求得:

<img src="https://latex.codecogs.com/png.latex?\underset{\tilde{P\epsilon&space;C}}{min}\&space;L(P,w)=\sum_{x,y}\tilde{P}(x,y)\sum_{i=1}^{n}w_if_i(x,y)-\sum_{x}\tilde{P}(x)logZ_w(x)" title="\underset{\tilde{P\epsilon C}}{min}\ L(P,w)=" />

故<img src="https://latex.codecogs.com/png.latex?\underset{\tilde{P\epsilon&space;C}}{min}\&space;L(P,w)=L_{\tilde{P}}(P_w)"  />

至此，证明了最大熵模型求解可以转换为极大对数似然估计。

# 最大熵模型和逻辑回归

最大熵模型可以推导成为逻辑回归，逻辑回归只是最大熵模型的一种特殊情况。

最大熵模型为：

<img src="https://latex.codecogs.com/png.latex?P(y|x)=\frac{1}{Z(x)}exp\left&space;(&space;\sum_{i=1}^{n}w_if_i(x,y)&space;\right&space;)" title="P_w(y|x)=\frac{1}{Z_w(x)}exp\left ( \sum_{i=1}^{n}w_if_i(x,y) \right )" />

其中<img src="https://latex.codecogs.com/png.latex?Z(x)=\sum_{y}exp\left&space;(&space;\sum_{i=1}^{n}w_if_i(x,y)\right&space;)" title="Z(x)=\sum_{y}exp\left ( \sum_{i=1}^{n}w_if_i(x,y)\right )" />

如果分类类别只有两种<img src="https://latex.codecogs.com/gif.latex?y\epsilon&space;\{y_0,y_1\}" title="y\epsilon \{y_1,y_2\}" />

只定义一个特征函数：

<img src="https://latex.codecogs.com/gif.latex?f(x,y)=\left\{\begin{matrix}&space;g(x),\&space;\&space;y=y_1\\&space;0,\&space;\&space;\&space;\&space;\&space;\&space;y=y_0&space;\end{matrix}\right." title="f(x,y)=\left\{\begin{matrix} g(x),\ \ y=y_1\\ 0,\ \ \ \ \ \ y=y_0 \end{matrix}\right." />

将特征函数代入到最大熵模型，求<img src="https://latex.codecogs.com/gif.latex?y=y_1" title="y=y_1" />时的结果：

<img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;P(y_1|x)=&\frac{exp\left(wf(x,y_1)\right)}{\sum_{y}exp\left(wf(x,y)\right)}\\&space;=&\frac{exp\left(wf(x,y_1)\right)}{exp\left(wf(x,y_0)\right)&plus;exp\left(wf(x,y_1)\right)}\\&space;=&\frac{exp\left(wf(x,y_1)\right)}{exp\left(0\right)&plus;exp\left(wf(x,y_1)\right)}\\&space;=&\frac{exp\left(wg(x)\right)}{1&plus;exp\left(wg(x)\right)}\\&space;=&\frac{1}{1&plus;exp\left(-wg(x)\right)}\\&space;\end{align*}" title="\begin{align*} P(y_1|x)=&\frac{exp\left(wf(x,y_1)\right)}{\sum_{y}exp\left(wf(x,y)\right)}\\ =&\frac{exp\left(wf(x,y_1)\right)}{exp\left(wf(x,y_0)\right)+exp\left(wf(x,y_1)\right)}\\ =&\frac{exp\left(wf(x,y_1)\right)}{exp\left(0\right)+exp\left(wf(x,y_1)\right)}\\ =&\frac{exp\left(wg(x)\right)}{1+exp\left(wg(x)\right)}\\ =&\frac{1}{1+exp\left(-wg(x)\right)}\\ \end{align*}" />

求<img src="https://latex.codecogs.com/gif.latex?y=y_0" title="y=y_1" />时的结果：

<img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;P(y_0|x)=&\frac{exp\left(wf(x,y_0)\right)}{\sum_{y}exp\left(wf(x,y)\right)}\\&space;=&\frac{exp\left(wf(x,y_0)\right)}{exp\left(wf(x,y_0)\right)&plus;exp\left(wf(x,y_1)\right)}\\&space;=&\frac{exp\left(0\right)}{exp\left(0\right)&plus;exp\left(wf(x,y_1)\right)}\\&space;=&\frac{1}{1&plus;exp\left(wg(x)\right)}\\&space;=&\frac{exp\left(-wg(x)\right)}{exp\left(-wg(x)\right)&plus;1}\\&space;=&\frac{exp\left(-wg(x)\right)&plus;1-1}{exp\left(-wg(x)\right)&plus;1}\\&space;=&1-\frac{1}{exp\left(-wg(x)\right)&plus;1}\\&space;=&1-P(y_1|x)&space;\end{align*}" title="\begin{align*} P(y_0|x)=&\frac{exp\left(wf(x,y_0)\right)}{\sum_{y}exp\left(wf(x,y)\right)}\\ =&\frac{exp\left(wf(x,y_0)\right)}{exp\left(wf(x,y_0)\right)+exp\left(wf(x,y_1)\right)}\\ =&\frac{exp\left(0\right)}{exp\left(0\right)+exp\left(wf(x,y_1)\right)}\\ =&\frac{1}{1+exp\left(wg(x)\right)}\\ =&\frac{exp\left(-wg(x)\right)}{exp\left(-wg(x)\right)+1}\\ =&\frac{exp\left(-wg(x)\right)+1-1}{exp\left(-wg(x)\right)+1}\\ =&1-\frac{1}{exp\left(-wg(x)\right)+1}\\ =&1-P(y_1|x) \end{align*}" />

于是得到了逻辑回归模型

以上。

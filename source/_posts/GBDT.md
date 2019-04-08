---
title: GBDT 梯度提升树
date: 2019-04-05 21:42:35
tags: ["Algorithm"]
categories: Technic
---

# 提升树引入

提升树相当于提升方法在[CART](https://dorianzi.github.io/2019/03/30/CART/#more)上的应用。
平常我们讲的提升树就是GBDT （Gradient Boosting Decision Tree），它是通过拟合损失函数的负梯度值在当前模型的值。注意这里我们不是拟合真实值，而是拟合梯度值，只是梯度跟真实值是有关系的

GBDT有分类和回归两个方向的应用，本文主要介绍<font size="4">GBDT 回归提升树</font>。

# 关于函数梯度

GBDT的提升是加法模型，它不是定义一个固定结构的函数，然后通过样本拟合更新它的参数。它是函数本身的累加：<img src="https://latex.codecogs.com/gif.latex?F_m(X)=F_{m-1}(X)&plus;\Delta&space;F(X)" title="f_m(X)=f_{m-1}(X)+\Delta f(X)" />。所以如果要更快逼近最后的函数，我们就需要在正确的方向上变化，这个“正确的方向”当然就是损失函数减少最快的方向。所以我们需要用损失函数<img src="https://latex.codecogs.com/gif.latex?L(F(x))" title="L(f(x))" />对函数<img src="https://latex.codecogs.com/gif.latex?F(x)" title="f(x)" />求导，求得的导数，就是接下来<img src="https://latex.codecogs.com/gif.latex?F(x)" title="F(x)" />需要弥补的方向。这时候用一个函数能<img src="https://latex.codecogs.com/gif.latex?\Delta&space;F(X)" title="f_m(X)=f_{m-1}(X)+\Delta f(X)" />去拟合刚才求得的导数，那么函数就可以更新为<img src="https://latex.codecogs.com/gif.latex?F_m(X)=F_{m-1}(X)&plus;\Delta&space;F(X)" title="f_m(X)=f_{m-1}(X)+\Delta f(X)" />了。

导数值跟损失函数的选择有关系。如果选择平方损失误差<img src="https://latex.codecogs.com/gif.latex?L(y_i,F(x_i))=\frac{1}{2}[y_i-F(x_i)]^2" title="L(y_i,F(x_i))=\frac{1}{2}[y_i-F(x_i)]^2" />，那么它的导数就是：

<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;L(y_i,F(x_i))}{\partial&space;F(x_i)}&space;=y_i-F(x_i)" title="\frac{\partial L(y_i,F(x_i))}{\partial F(x_i)} =y_i-F(x_i)" />

令人惊喜的是这正是真实值和估计值之间的残差。 BTW，上面之所以用了<img src="https://latex.codecogs.com/gif.latex?\frac{1}{2}" title="\frac{1}{2}" />是为了计算方便，常数项并不会影响平方损失误差，以及残差的比较。

下面我们介绍的就是基于平方损失误差（也就是基于残差弥补）的GBDT回归实例。

# 用实例讲解GBDT

有以下数据需要用回归，并要求平方损失误差小于0.2时，可以停止建树：

![](/uploads/boosting_tree_1.png)

<font size="4" color="Blue">第一棵树</font>

1） 遍历各个切分点s=1.5,2.5,...,9.5找到平方损失误差最小值的切分点：

比如s=1.5,分割成了两个子集：<img src="https://latex.codecogs.com/gif.latex?R_1=\{x|x<1.5\},\&space;R_2=\{x|x>1.5\}," title="R_1=\{x|x<1.5\},\ R_2=\{x|x>1.5\}," />

通过公式<img src="https://latex.codecogs.com/gif.latex?m(s)=\underset{c_1}{min}\sum_{x_i\epsilon&space;R_1}^{&space;}(y_i-c_1)^2&plus;\underset{c_2}{min}\sum_{x_i\epsilon&space;R_2}^{&space;}(y_i-c_2)^2" title="\underset{c_1}{min}\sum_{x_i\epsilon R_1}^{ }(y_i-c_1)^2+\underset{c_2}{min}\sum_{x_i\epsilon R_2}^{ }(y_i-c_2)^2" />求平方损失误差

而其中<img src="https://latex.codecogs.com/gif.latex?c_1,c_2" title="c_1,c_2" />为各自子集的平均值<img src="https://latex.codecogs.com/gif.latex?c_1=\frac{1}{N_1}\sum_{x_i\epsilon&space;R_1}^{&space;}y_i,\&space;c_2=\frac{1}{N_2}\sum_{x_i\epsilon&space;R_1}^{&space;}y_i" title="c_1=\frac{1}{N_1}\sum_{x_i\epsilon R_1}^{ }y_i,\ c_2=\frac{1}{N_2}\sum_{x_i\epsilon R_1}^{ }y_i" />时，可以使得每个子集的平方损失误差最小。

求平均值为：<img src="https://latex.codecogs.com/gif.latex?c_1=5.56,\&space;c_2=7.50" title="c_1=5.56,\ c_2=7.50" />，进而求得平方损失误差为<img src="https://latex.codecogs.com/gif.latex?m(1.5)=15.72" title="m(1.5)=15.72" />

同样的方法求得其它切分点的平方损失误差，列表入下：

![](/uploads/boosting_tree_2.png)

可见，当s=6.5时, <img src="https://latex.codecogs.com/gif.latex?m(6.5)=1.93" title="m(6.5)=1.93" />为所有切分点里平方损失误差最小的

2) 选择切分点s=6.5构建第一颗回归树，各分支数值使用<img src="https://latex.codecogs.com/gif.latex?c_1=6.24,\&space;c_2=8.91" title="c_1=6.24,\ c_2=8.91" />：

<img src="https://latex.codecogs.com/gif.latex?T_1(x)=&space;\left\{\begin{matrix}&space;6.24,\&space;\&space;x<6.5\\&space;8.91,\&space;\&space;x\geq&space;6.5&space;\end{matrix}\right." title="T_1(x)= \left\{\begin{matrix} 6.24,\ \ x<6.5\\ 8.91,\ \ x\geq 6.5 \end{matrix}\right." />

第一轮过后，我们提升树为:

<img src="https://latex.codecogs.com/gif.latex?f_1(x)=T_1(x)" title="f_1(x)=T_1(x)" />

3) 求提升树拟合数据的残差和平方损失误差：

提升树拟合数据的残差计算：<img src="https://latex.codecogs.com/gif.latex?r_{2i}=y_i-f_1(x_i),\&space;\&space;i=1,2,...,10" title="r_2i=y_i-f_1(x_i),\ \ i=1,2,...,10" />

各个点的计算结果：

![](/uploads/boosting_tree_3.png)

提升树拟合数据的平方损失误差计算：

<img src="https://latex.codecogs.com/gif.latex?L(y,f_1(x))=\sum_{i=1}^{10}(y_i-f_1(x_i))^2=1.93" title="L(y,f_1(x))=\sum_{i=1}^{10}(y_i-f_1(x_i))^2=1.93" />

大于0.2，则还需要继续建树。

<font size="4" color="Blue">第二棵树</font>

4) 确定需要拟合的训练数据为上一棵树的残差：

![](/uploads/boosting_tree_3.png)

5） 遍历各个切分点s=1.5,2.5,...,9.5找到平方损失误差最小值的切分点：

同样的方法求得其它切分点的平方损失误差，列表入下：

![](/uploads/boosting_tree_4.png)

可见，当s=3.5时, <img src="https://latex.codecogs.com/gif.latex?m(3.5)=0.79" />为所有切分点里平方损失误差最小的

6) 选择切分点s=3.5构建第二颗回归树，各分支数值使用<img src="https://latex.codecogs.com/gif.latex?c_1=-0.52,\&space;c_2=0.22"  /> ：

<img src="https://latex.codecogs.com/gif.latex?T_2(x)=&space;\left\{\begin{matrix}&space;-0.52,\&space;\&space;x<3.5\\&space;0.22,\&space;\&space;\&space;\&space;x\geq&space;0.5&space;\end{matrix}\right."  />

第二轮过后，我们提升树为:

<img src="https://latex.codecogs.com/gif.latex?f_2(x)=f_1(x)&plus;T_2(x)=\left\{\begin{matrix}&space;5.72,\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;x<3.5\\&space;6.46,\&space;\&space;3.5\leq&space;x<6.5\\&space;9.13,\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;x\geq&space;6.5&space;\end{matrix}\right."  />

7) 求提升树拟合数据的残差和平方损失误差：

提升树拟合数据的残差计算：<img src="https://latex.codecogs.com/gif.latex?r_{3i}=y_i-f_2(x_i),\&space;\&space;i=1,2,...,10"  />

各个点的计算结果，同时对比初始值和上一颗树的残差：

![](/uploads/boosting_tree_5.png)

可以看见，随着树的增多，残差一直在减少。

到目前为止，提升树拟合数据的平方损失误差计算：

<img src="https://latex.codecogs.com/gif.latex?L(y,f_2(x))=\sum_{i=1}^{10}(y_i-f_2(x_i))^2=0.79"  />

多说一句，这里是从全局提升树的角度去计算损失，其实和上面第5）步中从最后一颗树的角度去计算损失，结果是一样的

目前损失大于0.2的阈值，还需要继续建树

<font size="4" color="Blue">...</font> &emsp;

<font size="4" color="Blue">...</font> &emsp;

<font size="4" color="Blue">第六棵树</font>

到第六颗树的时候，我们已经累计获得了：

<img src="https://latex.codecogs.com/gif.latex?T_3(x)=\left\{\begin{matrix}&space;0.15,\&space;\&space;\&space;\&space;x<6.5\\&space;-0.22,\&space;\&space;x\geq&space;6.5&space;\end{matrix}\right." /> &emsp;&emsp; <img src="https://latex.codecogs.com/gif.latex?T_4(x)=\left\{\begin{matrix}&space;-0.16,\&space;\&space;x<4.5\\&space;0.11,\&space;\&space;\&space;\&space;\&space;x\geq&space;4.5&space;\end{matrix}\right." title="T_4(x)=\left\{\begin{matrix} -0.16,\ \ x<4.5\\ 0.11,\ \ \ \ \ x\geq 4.5 \end{matrix}\right." />

<img src="https://latex.codecogs.com/gif.latex?T_5(x)=\left\{\begin{matrix}&space;0.07,\&space;\&space;\&space;\&space;x<6.5\\&space;-0.11,\&space;\&space;x\geq&space;6.5&space;\end{matrix}\right." /> &emsp;&emsp; <img src="https://latex.codecogs.com/gif.latex?T_5(x)=\left\{\begin{matrix}&space;-0.15,\&space;\&space;x<2.5\\&space;0.04,\&space;\&space;\&space;\&space;\&space;x\geq&space;2.5&space;\end{matrix}\right." />

此时提升树为：

<img src="https://latex.codecogs.com/gif.latex?f_6(x)=T_1(x)&plus;T_2(x)&plus;...&plus;T_6(x)\\&space;=\left\{\begin{matrix}&space;5.63,\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;x<2.5\\&space;5.82,\&space;\&space;2.5\leq&space;x<3.5\\&space;6.56,\&space;\&space;3.5\leq&space;x<4.5\\&space;6.83,\&space;\&space;4.5\leq&space;x<6.5\\&space;8.95,\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;x\geq&space;6.5&space;\end{matrix}\right."  />

此时用<img src="https://latex.codecogs.com/gif.latex?f_6(x)"  />拟合训练数据的平方损失误差为：

<img src="https://latex.codecogs.com/gif.latex?L(y,f_6(x))=\sum_{i=1}^{10}(y_i-f_6(x_i))^2=0.17"  />

平方损失误差小于0.2的阈值，停止建树。

<img src="https://latex.codecogs.com/gif.latex?f_6(x)"  /> 为我们最终所求的提升树。

以上。


# 参考

https://blog.csdn.net/qq_22238533/article/details/79199605
http://docs.salford-systems.com/GreedyFuncApproxSS.pdf


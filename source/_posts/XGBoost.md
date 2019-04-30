---
title: XGBoost 提升树的高效实现
date: 2019-04-14 20:33:01
tags: ["Algorithm"]
categories: Technic
---

eXtreme Gradient Boosting简称XGBoost，不算一种全新的算法，而是GBDT的一种高效的实现，它的repo在[这里](https://github.com/dmlc/xgboost)

XGBoost在[GBDT](https://dorianzi.github.io/2019/04/05/GBDT/)基础上进行了很多抽象和优化，以至于乍一看不像同一个算法了。接下来解释XGBoost的原理

# XGBoost原理推导

首先它继承的是GBDT的Boosting思想，也就是生成多颗树，每棵树都有对某个样本的预测，用加法模型将多颗树对该样本的预测结果相加，就是该样本的最终预测结果。而损失函数和GBDT一样，都是对所有样本预测值和label值偏差的求和，这个偏差可以是均方误差也可以是别的误差。不同的是XGBoost抽象出了损失函数并且加上了惩罚项：

<img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;Obj=&\sum_{i=1}^{n}l(y_i,\hat{y_i})&plus;\sum_{k=1}^{K}\Omega(f_k)\\&space;=&\sum_{i=1}^{n}l(y_i,\hat{y_i})&plus;\gamma&space;T&plus;\lambda\frac{1}{2}\sum_{j=1}^{T}w^{2}_j&space;\end{align*}" />

其中<img src="https://latex.codecogs.com/gif.latex?\sum_{i=1}^{n}l(y_i,\hat{y_i})" />为GBDT的损失函数，是样本偏差之和，<img src="https://latex.codecogs.com/gif.latex?n" title="n" />为样本数，<img src="https://latex.codecogs.com/gif.latex?l" title="l" />为偏差函数，比如均方误差，<img src="https://latex.codecogs.com/gif.latex?y_i" title="y_i" />为样本的label值，<img src="https://latex.codecogs.com/gif.latex?\hat{y_i}" title="\hat{y_i}" />为预测值。

<img src="https://latex.codecogs.com/gif.latex?\sum_{k=1}^{K}\Omega(f_k)=\gamma&space;T&plus;\lambda\frac{1}{2}\sum_{j=1}^{T}w^{2}_j" title="\sum_{k=1}^{K}\Omega(f_k)=\gamma T+\lambda\frac{1}{2}\sum_{j=i}^{T}w^{2}_j" />  为惩罚项，<img src="https://latex.codecogs.com/gif.latex?K"/>为树的总棵树，<img src="https://latex.codecogs.com/gif.latex?T"/>为所有树的叶子总个数，<img src="https://latex.codecogs.com/gif.latex?w_j" title="w_j" />为第j个叶子的值，这里用的是它的二范数。<img src="https://latex.codecogs.com/gif.latex?\gamma&space;,\lambda" title="\gamma ,\lambda" />是乘数因子。

可以看出惩罚项惩罚的是总叶子的个数，相当于也惩罚了树的颗数。以此降低过拟合风险。

接下来把注意力放到住损失函数上来。现在想象我们已经生成到了第t一棵树，我们当然希望，这第t颗树的生成使得总损失减少了，不然就没必要生成第t棵树了。

设第t棵树的预测对样本<img src="https://latex.codecogs.com/gif.latex?x_i" title="x_i" />的预测函数为<img src="https://latex.codecogs.com/gif.latex?f_t(x_i)"  />，那么样本的最终预测值是所有树的预测值之和，也是前t-1棵树的最终预测值加当前树的单独预测值：

<img src="https://latex.codecogs.com/gif.latex?\hat{y}^{(t)}_{i}=\hat{y}^{(t-1)}_{i}&plus;f_t(x_i)" title="\hat{y}^{(t)}_{i}=\hat{y}^{(t-1)}_{i}+f_t(x_i)" />

代入到损失函数中，得到t颗数总损失为:

<img src="https://latex.codecogs.com/gif.latex?&space;L^{(t)}=&\sum_{i=1}^{n}l(y_i,\hat{y}^{(t-1)}_{i}&plus;f_t(x_i))&plus;\sum_{k=1}^{K}\Omega(f_k)" />

接下来巧妙的来了：将函数<img src="https://latex.codecogs.com/gif.latex?l" title="l" />在点<img src="https://latex.codecogs.com/gif.latex?\hat{y}^{(t-1)}_i"  />处按<font size="4">泰勒二次展开</font>，并去掉常数项：

<img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;\tilde{L}^{(t)}=&\sum_{i=1}^{n}[l(y_i,\hat{y}^{(t-1)}_{i})&plus;g_if_t(x_i)&plus;\frac{1}{2}h_if^2_t(x_i)]&plus;\sum_{k=1}^{K}\Omega(f_k)\\&space;=&\sum_{i=1}^{n}[g_if_t(x_i)&plus;\frac{1}{2}h_if^2_t(x_i)]&plus;\gamma&space;T&plus;\lambda\frac{1}{2}\sum_{j=1}^{T}w^{2}_j&space;\end{align*}"  />

其中<img src="https://latex.codecogs.com/gif.latex?g_i" title="g_i" />为偏差函数<img src="https://latex.codecogs.com/gif.latex?l" title="l" />在点<img src="https://latex.codecogs.com/gif.latex?\hat{y}^{(t-1)}_i"  />处的一阶导数，<img src="https://latex.codecogs.com/gif.latex?h_i"  />为偏差函数<img src="https://latex.codecogs.com/gif.latex?l" title="l" />在点<img src="https://latex.codecogs.com/gif.latex?\hat{y}^{(t-1)}_i"  />处的二阶导数

我们回顾一下[GBDT的梯度](https://dorianzi.github.io/2019/04/05/GBDT/#%E5%85%B3%E4%BA%8E%E5%87%BD%E6%95%B0%E6%A2%AF%E5%BA%A6)，它只用到了对偏差函数一阶导数。所以说这里，使用二阶导数是XGBoost的一个优化，这样使得它能够使用任何损失函数，比如最小二乘函数([参考](https://www.zhihu.com/question/61374305))，以此提高扩展性。

注意，上式之所以可以去掉常数项，是因为损失函数作为目标函数，进行最小化的过程中，常数项不影响求导结果，所以不影响目标参数的求解结果。被去掉常数项的损失函数的第一项求和部分<img src="https://latex.codecogs.com/gif.latex?\sum_{i=1}^{n}[g_if_t(x_i)&plus;\frac{1}{2}h_if^2_t(x_i)]"  />可以理解为第t棵树里每个样本的损失之和。

下面，欲将以上目标函数写成按照叶子节点求和的形式。首先设<img src="https://latex.codecogs.com/gif.latex?f_t(x)=w_{q(x)}" title="f_t(x)=w_{q(x)}" />，表示是总叶子数中第<img src="https://latex.codecogs.com/gif.latex?q(x)" title="q(x)" />个叶子的值，而<img src="https://latex.codecogs.com/gif.latex?q(x)" title="q(x)" />是落入第t棵树的。在一棵树里，n个总样本中，总是多个样本为一组，落入每个叶子中，并共享同一个值，每棵树都如此，只是每棵树的样本分组方式不一样。同时，第一项求和部分为第t棵树里每个样本的损失之和。考虑上述形式，将目标函数改写为：

<img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;\tilde{L}^{(t)}=&\sum_{i=1}^{n}[g_iw_{q(x_i)}&plus;\frac{1}{2}h_iw^2_{q(x_i)}]&plus;\gamma&space;T&plus;\lambda\frac{1}{2}\sum_{j=1}^{T}w^{2}_j\\&space;=&\sum_{j=1}^{T}[(\sum_{i\epsilon&space;I_j}g_i)w_j&plus;\frac{1}{2}(\sum_{i\epsilon&space;I_j}h_i&plus;\lambda)w^2_j]&plus;\gamma&space;T&space;\end{align*}"  />

其中<img src="https://latex.codecogs.com/gif.latex?I_j}"  />表示第j个叶子里落入的那组样本的下标，它们是共享同一个值<img src="https://latex.codecogs.com/gif.latex?w_j}" />的。 令<img src="https://latex.codecogs.com/gif.latex?G_j=\sum_{i\epsilon&space;I_j}g_i,\&space;H_j=\sum_{i\epsilon&space;I_j}h_i" title="G_j=\sum_{i\epsilon I_j}g_i,\ H_j=\sum_{i\epsilon I_j}h_i" />，则:

<img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;\tilde{L}^{(t)}=&\sum_{j=1}^{T}[G_jw_j&plus;\frac{1}{2}(H_j&plus;\lambda)w^2_j]&plus;\gamma&space;T&space;\end{align*}" />

要求的是<img src="https://latex.codecogs.com/gif.latex?w_j" title="w_j" />，所以对它求偏导为0，得：

<img src="https://latex.codecogs.com/gif.latex?G_j&plus;(H_j&plus;\lambda)w_j=0" title="G_j+(H_j+\lambda)w_j=0" />

求解得：

<img src="https://latex.codecogs.com/gif.latex?w_j^{*}=-\frac{G_j}{H_j&plus;\lambda&space;}" title="w_j^{*}=-\frac{G_j}{H_j+\lambda }" />

至此我们可以通过第t-1棵树的导数，来求各叶子<img src="https://latex.codecogs.com/gif.latex?w_j" title="w_j" />的数值，包括第t棵树的叶子的值。

同时我们也可以得到目标函数的值,在节点分裂效果评估中会用到：

<img src="https://latex.codecogs.com/gif.latex?\tilde{L}^{(t)}=-\frac{1}{2}\sum_{j=1}^{T}\frac{G^2_j}{H_j&plus;\lambda}&plus;\gamma&space;T" title="\tilde{L}^{(t)}=-\frac{1}{2}\sum_{j=1}^{T}\frac{G^2_j}{H_j+\lambda}+\gamma T" />

图示计算样例：

![](/uploads/XGBoost_1.png)

# XGBoost的节点分裂

上一节只是基于树结构确定的情况下的叶子值求解算法，然而作为前提，树的特征分裂方法，还没有被讨论到。这一节就来讨论它。
XGBoost采用的方法是遍历所有特征的分裂点。如何评价分裂效果？使用上述的目标函数。只要逐一计算每种分裂方法下上述的目标函数的值，越小的分裂效果越好。当然不能无限制分裂，为了限制树生长过深，还加了个阈值，只有当增益大于该阈值才进行分裂：

![](/uploads/XGBoost_2.png)

虽然需要贪心地计算很多遍分裂，但是好在特征的分裂是能够<font size="4">并行</font>的

# XGBoost的其它优化

时刻记住XGBoost是重在实现，它是个工具，提供了很多优化方式，在编程上，可以使用也可以不使用。

如针对稀疏数据的算法（缺失值处理）：当样本的第i个特征值缺失时，无法利用该特征进行划分时，XGBoost的想法是将该样本分别划分到左结点和右结点，然后计算其增益，哪个大就划分到哪边。

如防止过拟合的Shrinkage方法：在每次迭代中对树的每个叶子值乘上一个缩减权重η，减小该树的比重，让后面的树有优化的空间。

又如防止过拟合的Column Subsampling方法：类似于随机森林，选取部分特征建树。可分为两种：1）按层随机采样，在对同一层内每个结点分裂之前，先随机选择一部分特征，然后只需要遍历这部分的特征，来确定最优的分割点；2）随机选择特征，则建树前随机选择一部分特征然后分裂就只遍历这些特征。一般情况下前者效果更好。

还有其它优化方式，不逐一讲解了：交叉验证，early stop，支持设置样本权重，支持并行化等。

# 参考

https://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf


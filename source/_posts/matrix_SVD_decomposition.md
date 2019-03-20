---
title: SVD Decomposition
date: 2019-03-09 02:00:00
tags: "Algorithm"
---

SVD分解即奇异值分解， 可以从特征值分解推导而来。先理解特征值分解

# 特征值和特征向量
对矩阵A，存在特征向量<img src="https://latex.codecogs.com/gif.latex?\alpha" title="\alpha"  style="display:inline;vertical-align:text-top;"/>和特征值<img src="https://latex.codecogs.com/gif.latex?\lambda" title="\lambda"  style="display:inline;vertical-align:text-top;"/>满足：<img src="https://latex.codecogs.com/gif.latex?A\alpha=\lambda\alpha" title="A\alpha=\lambda\alpha"  style="display:inline;vertical-align:text-top;"/>

如果把矩阵A理解为线性变换，那么上式表示：可以找到向量<img src="https://latex.codecogs.com/gif.latex?\alpha" title="\alpha"  style="display:inline;vertical-align:text-top;"/>使得A只能对它进行<img src="https://latex.codecogs.com/gif.latex?\lambda" title="\lambda"  style="display:inline;vertical-align:text-top;"/>倍的拉伸。

以<font size=5>基变换</font>来理解。我们先构建一个“绝对”坐标系（如下图）。为了便于计算，设置原基为<img src="https://latex.codecogs.com/gif.latex?\binom{1}{0}" title="\binom{1}{0}"  style="display:inline;vertical-align:text-top;"/>和<img src="https://latex.codecogs.com/gif.latex?\binom{0}{1}" title="\binom{0}{1}"  style="display:inline;vertical-align:text-top;"/>，正好与“绝对”坐标系重叠。

通过矩阵<img src="https://latex.codecogs.com/gif.latex?\begin{bmatrix}&space;3&&space;1\\&space;0&2&space;\end{bmatrix}" title="\begin{bmatrix} 3& 1\\ 0&2 \end{bmatrix}"  style="display:inline;vertical-align:text-top;"/>换基，可直接计算得到实际上是将原基变换为新基<img src="https://latex.codecogs.com/gif.latex?\binom{3}{0}" title="\binom{3}{0}"  style="display:inline;vertical-align:text-top;"/>和<img src="https://latex.codecogs.com/gif.latex?\binom{1}{2}" title="\binom{1}{2}"  style="display:inline;vertical-align:text-top;"/>：

<img src="https://latex.codecogs.com/gif.latex?\begin{bmatrix}&space;3&space;&&space;1\\&space;0&space;&&space;2&space;\end{bmatrix}\begin{bmatrix}&space;1\\&space;0&space;\end{bmatrix}=\begin{bmatrix}&space;3\\&space;0&space;\end{bmatrix}" title="\begin{bmatrix} 3 & 1\\ 0 & 2 \end{bmatrix}\begin{bmatrix} 1\\ 0 \end{bmatrix}=\begin{bmatrix} 3\\ 0 \end{bmatrix}"  style="display:inline;vertical-align:text-top;"/>

<img src="https://latex.codecogs.com/gif.latex?\begin{bmatrix}&space;3&space;&&space;1\\&space;0&space;&&space;2&space;\end{bmatrix}\begin{bmatrix}&space;0\\&space;1&space;\end{bmatrix}=\begin{bmatrix}&space;1\\&space;2&space;\end{bmatrix}" title="\begin{bmatrix} 3 & 1\\ 0 & 2 \end{bmatrix}\begin{bmatrix} 0\\ 1 \end{bmatrix}=\begin{bmatrix} 1\\ 2 \end{bmatrix}"  style="display:inline;vertical-align:text-top;"/>

换基不影响坐标值，所以原基下的坐标(1,1)，变换后在新基下的坐标也是(1,1），而在“绝对”坐标系中坐标为(4,2)

<img src="https://latex.codecogs.com/gif.latex?\begin{bmatrix}&space;3&space;&&space;1\\&space;0&space;&&space;2&space;\end{bmatrix}\begin{bmatrix}&space;1\\&space;1&space;\end{bmatrix}=\begin{bmatrix}&space;4\\&space;2&space;\end{bmatrix}" title="\begin{bmatrix} 3 & 1\\ 0 & 2 \end{bmatrix}\begin{bmatrix} 1\\ 1 \end{bmatrix}=\begin{bmatrix} 4\\ 2 \end{bmatrix}"  style="display:inline;vertical-align:text-top;"/>

<img src="https://github.com/DorianZi/algorithm_explained/raw/master/res/pic_3.png" style="display:inline;vertical-align:text-top;">

在基变换体系下，我们无法直接获得特征向量和特征值的几何意义，以空间拉伸来解释会获得更直观理解。

以<font size=5>空间拉伸</font>来理解，以下线性变换将正方形围住的空间变换成普通平行四边形。大部分向量都被施加了旋转和拉伸（<font color=#8B0000>褐色</font>），只有特征向量<img src="https://latex.codecogs.com/gif.latex?\alpha_{1},\alpha_{2}" title="\alpha_{1},\alpha_{2}"  style="display:inline;vertical-align:text-top;"/>（<font color=#008000>绿色</font>和<font color=#0000FF>蓝色</font>）所在方向的向量只进行了拉伸，且拉伸效果为相应的特征值<img src="https://latex.codecogs.com/gif.latex?\lambda_{1},\lambda_{2}" title="\lambda_{1},\lambda_{2}"  style="display:inline;vertical-align:text-top;"/>

<img src="https://github.com/DorianZi/algorithm_explained/raw/master/res/pic1.png" style="display:inline;vertical-align:text-top;">


再回到最上面的图,如果继续不断地对空间施加矩阵A的线性变换(<img src="https://latex.codecogs.com/gif.latex?A^{n}" title="A^{n}"  style="display:inline;vertical-align:text-top;"/>), 那么特征向量所在方向的向量（<font color=#008000>绿色</font>和<font color=#0000FF>蓝色</font>）会幂级拉伸，非特征向量所以在方向的向量（<font color=#8B0000>褐色</font>）会越来越<font size=5>接近</font>特征向量方向。特殊情况，当<img src="https://latex.codecogs.com/gif.latex?\lambda=1" title="\lambda=1"  style="display:inline;vertical-align:text-top;"/>时，系统会<font size=5>稳定</font>在<font size=5>最大</font>特征值对应的特征向量方向。
<img src="https://github.com/DorianZi/algorithm_explained/raw/master/res/pic4.png" style="display:inline;vertical-align:text-top;">

# 矩阵对角化
将矩阵对角化为特征向量为列的矩阵<img src="https://latex.codecogs.com/gif.latex?P" title="P"  style="display:inline;vertical-align:text-top;"/>和对角矩阵<img src="https://latex.codecogs.com/gif.latex?\Lambda" title="\Lambda"  style="display:inline;vertical-align:text-top;"/>的乘积：

<img src="https://latex.codecogs.com/gif.latex?A=P\Lambda&space;P^{-1}" title="A=P\Lambda P^{-1}"  style="display:inline;vertical-align:text-top;"/>

其中<img src="https://latex.codecogs.com/gif.latex?P" title="P"  style="display:inline;vertical-align:text-top;"/>的列向量为特征向量，<img src="https://latex.codecogs.com/gif.latex?\Lambda" title="\Lambda"  style="display:inline;vertical-align:text-top;"/>为特征值矩阵。

## 数学推导
对角化推导如下：

根据特征值/特征向量定义：

<img src="https://latex.codecogs.com/gif.latex?A\alpha_{1}=\lambda_{1}\alpha_{1},A\alpha_{2}=\lambda_{2}\alpha_{2},...,A\alpha_{n}=\lambda_{n}\alpha_{n}" title="A\alpha_{1}=\lambda_{1}\alpha_{1},A\alpha_{2}=\lambda_{2}\alpha_{2},...,A\alpha_{n}=\lambda_{n}\alpha_{n}"  style="display:inline;vertical-align:text-top;"/>

合并为矩阵形式：

<img src="https://latex.codecogs.com/gif.latex?A(\alpha_{1},\alpha_{2},...,\alpha_{n})&space;=&space;(\lambda_{1}\alpha_{1},\lambda_{2}\alpha_{2},...,\lambda_{n}\alpha_{n})" title="A(\alpha_{1},\alpha_{2},...,\alpha_{n}) = (\lambda_{1}\alpha_{1},\lambda_{2}\alpha_{2},...,\lambda_{n}\alpha_{n})"  style="display:inline;vertical-align:text-top;"/>

<img src="https://latex.codecogs.com/gif.latex?A(\alpha_{1},\alpha_{2},...,\alpha_{n})&space;=&space;(\alpha_{1},\alpha_{2},...,\alpha_{n})&space;\begin{bmatrix}&space;\lambda_{1}&&space;&&space;&&space;\\&space;&&space;\lambda_{2}&&space;&&space;\\&space;&&space;&&space;...&&space;\\&space;&&space;&&space;&\lambda_{n}&space;\end{bmatrix}" title="A(\alpha_{1},\alpha_{2},...,\alpha_{n}) = (\alpha_{1},\alpha_{2},...,\alpha_{n}) \begin{bmatrix} \lambda_{1}& & & \\ & \lambda_{2}& & \\ & & ...& \\ & & &\lambda_{n} \end{bmatrix}"  style="display:inline;vertical-align:text-top;"/>

<img src="https://latex.codecogs.com/gif.latex?A=(\alpha_{1},\alpha_{2},...,\alpha_{n})&space;\begin{bmatrix}&space;\lambda_{1}&&space;&&space;&&space;\\&space;&&space;\lambda_{2}&&space;&&space;\\&space;&&space;&&space;...&&space;\\&space;&&space;&&space;&\lambda_{n}&space;\end{bmatrix}&space;(\alpha_{1},\alpha_{2},...,\alpha_{n})^{-1}" title="A=(\alpha_{1},\alpha_{2},...,\alpha_{n}) \begin{bmatrix} \lambda_{1}& & & \\ & \lambda_{2}& & \\ & & ...& \\ & & &\lambda_{n} \end{bmatrix} (\alpha_{1},\alpha_{2},...,\alpha_{n})^{-1}"  style="display:inline;vertical-align:text-top;"/>

## 几何推导
前提知识：从新的一组基<img src="https://latex.codecogs.com/gif.latex?(\alpha_{1},\alpha_{2},..,\alpha_{n})" title="(\alpha_{1},\alpha_{2},..,\alpha_{n})"  style="display:inline;vertical-align:text-top;"/>的角度来看待一个线性变换A为：<img src="https://latex.codecogs.com/gif.latex?(\alpha_{1},\alpha_{2},...,\alpha_{n})^{-1}A(\alpha_{1},\alpha_{2},...,\alpha_{n})" title="(\alpha_{1},\alpha_{2},...,\alpha_{n})^{-1}A(\alpha_{1},\alpha_{2},...,\alpha_{n})"  style="display:inline;vertical-align:text-top;"/>

那么基于这个前提知识，我们抽取出特征向量组成一组新基<img src="https://latex.codecogs.com/gif.latex?(\alpha_{1},\alpha_{2},..,\alpha_{n})" title="(\alpha_{1},\alpha_{2},..,\alpha_{n})"  style="display:inline;vertical-align:text-top;"/>, 从这组新基的角度来看待线性变换A为等式左边；等式右边一定为对角矩阵，且对角元素值为特征值。这是因为从特征向量的角度看来，A只是完成了拉伸，而没有旋转：

<img src="https://latex.codecogs.com/gif.latex?(\alpha_{1},\alpha_{2},...,\alpha_{n})^{-1}A(\alpha_{1},\alpha_{2},...,\alpha_{n})=\begin{bmatrix}&space;\lambda_{1}&space;&&space;&&space;&&space;\\&space;&&space;\lambda_{2}&space;&&space;&&space;\\&space;&&space;&&space;...&&space;\\&space;&&space;&&space;&&space;\lambda_{n}&space;\end{bmatrix}" title="(\alpha_{1},\alpha_{2},...,\alpha_{n})^{-1}A(\alpha_{1},\alpha_{2},...,\alpha_{n})=\begin{bmatrix} \lambda_{1} & & & \\ & \lambda_{2} & & \\ & & ...& \\ & & & \lambda_{n} \end{bmatrix}"  style="display:inline;vertical-align:text-top;"/>

另外，从过程上理解，对任意一个向量<img src="https://latex.codecogs.com/gif.latex?\beta" title="\beta"  style="display:inline;vertical-align:text-top;"/>施加矩阵A变换，相当于连续施加三个变换:<img src="https://latex.codecogs.com/gif.latex?P^{-1}" title="P^{-1}"  style="display:inline;vertical-align:text-top;"/> -> <img src="https://latex.codecogs.com/gif.latex?\Lambda" title="\Lambda"  style="display:inline;vertical-align:text-top;"/> -> <img src="https://latex.codecogs.com/gif.latex?P" title="P"  style="display:inline;vertical-align:text-top;"/> :

<img src="https://latex.codecogs.com/gif.latex?A\beta=P\Lambda&space;P^{-1}\beta" title="A\beta=P\Lambda P^{-1}\beta"  style="display:inline;vertical-align:text-top;"/>


特殊情况下，如果<img src="https://latex.codecogs.com/gif.latex?P" title="P"  style="display:inline;vertical-align:text-top;"/>和<img src="https://latex.codecogs.com/gif.latex?P^{-1}" title="P^{-1}"  style="display:inline;vertical-align:text-top;"/>为正交矩阵，则只有旋转效果,且它们互为反向旋转；<img src="https://latex.codecogs.com/gif.latex?\Lambda" title="\Lambda"  style="display:inline;vertical-align:text-top;"/>为对角矩阵，只有拉伸效果。所以改变换为:

旋转 -> 拉伸 -> 转回来 


## SVD奇异值分解
SVD = Singular Value Decomposition 即奇异值分解。特征值可以理解为奇异值的一种特殊情况，即矩阵为方阵的情况。在矩阵非方阵的时候，我们将类似的分解称作奇异值分解

我们有一个非方阵的维度为<img src="https://latex.codecogs.com/gif.latex?m\times&space;n(m<n)" title="m\times n(m<n)"  style="display:inline;vertical-align:text-top;"/>的矩阵<img src="https://latex.codecogs.com/gif.latex?\underset{m\times&space;n}{A}" title="\underset{m\times n}{A}"  style="display:inline;vertical-align:text-top;"/>，希望能够对角化。采用上面的特征矩阵分解方式是不可能了，但是我们可以利用<img src="https://latex.codecogs.com/gif.latex?AA^{T}" title="AA^{T}"  style="display:inline;vertical-align:text-top;"/>和<img src="https://latex.codecogs.com/gif.latex?A^{T}A" title="A^{T}A"  style="display:inline;vertical-align:text-top;"/>为实对称方阵的特性来获得不同的特征矩阵分解：

<img src="https://latex.codecogs.com/gif.latex?\underset{m\times&space;n}{A}\underset{n\times&space;m}{A^{T}}=\underset{m\times&space;m}{U}\underset{m\times&space;m}{\Lambda&space;_{2}}\underset{m\times&space;m}{U^{-1}}=\underset{m\times&space;m}{U}\underset{m\times&space;m}{\Lambda&space;_{2}}\underset{m\times&space;m}{U^{T}}">

<img src="https://latex.codecogs.com/gif.latex?\underset{n\times&space;m}{A^{T}}\underset{m\times&space;n}{A}=\underset{n\times&space;n}{V}\underset{n\times&space;n}{\Lambda&space;_{1}}\underset{n\times&space;n}{V^{-1}}=\underset{n\times&space;n}{V}\underset{n\times&space;n}{\Lambda&space;_{1}}\underset{n\times&space;n}{V^{T}}">
可以推得(推导过程TBD)：

<img src="https://latex.codecogs.com/gif.latex?\underset{m\times&space;n}{A}=\underset{m\times&space;m}{U}\underset{m\times&space;n}{\sum}\underset{n\times&space;n}{V^{T}}" style="display:inline;vertical-align:text-top;">

<img src="https://github.com/DorianZi/algorithm_explained/blob/master/res/SVD_graph.png?raw=true" style="display:inline;vertical-align:text-top;">


### 奇异值求解推导
接下来我们来求新的对角矩阵<img src="https://latex.codecogs.com/gif.latex?\underset{m\times&space;n}{\sum}" title="\underset{m\times n}{\sum}"  style="display:inline;vertical-align:text-top;"/>的对角元素，即奇异值<img src="https://latex.codecogs.com/gif.latex?\lambda_{1},\lambda_{2},...,\lambda_{n}" title="\lambda_{1},\lambda_{2},...,\lambda_{n}"  style="display:inline;vertical-align:text-top;"/>

首先了解一下<img src="https://latex.codecogs.com/gif.latex?\underset{m\times&space;n}{\sum}" title="\underset{m\times n}{\sum}"  style="display:inline;vertical-align:text-top;"/>的形状：

<img src="https://latex.codecogs.com/gif.latex?\underset{m\times&space;n}{\sum&space;}=&space;\underset{m\times&space;n}{\begin{bmatrix}&space;\lambda_{1}&&space;&&space;&&space;&&space;0\\&space;&&space;\lambda_{2}&&space;&&space;&&space;\\&space;&&space;&&space;...&&space;&&space;\\&space;&&space;&&space;&&space;&&space;\lambda_{n}\\&space;&&space;&&space;&&space;&&space;0\\&space;&&space;&&space;&&space;&&space;...\\&space;0&space;&&space;&&space;&&space;&&space;0\\&space;\end{bmatrix}&space;}" title="\underset{m\times n}{\sum }= \underset{m\times n}{\begin{bmatrix} \lambda_{1}& & & & 0\\ & \lambda_{2}& & & \\ & & ...& & \\ & & & & \lambda_{n}\\ & & & & 0\\ & & & & ...\\ 0 & & & & 0\\ \end{bmatrix} }"  style="display:inline;vertical-align:text-top;"/>

开始推导：

<img src="https://latex.codecogs.com/gif.latex?A=U\sum&space;V^{T}\&space;\&space;\&space;\&space;\&space;\&space;=>\&space;\&space;\&space;\&space;\&space;\&space;AV=U\sum&space;V^{T}V&space;\&space;\&space;\&space;\&space;\&space;\underset{=======>}{V^{T}V=I}&space;\&space;\&space;\&space;\&space;AV=U\sum" title="A=U\sum V^{T}\ \ \ \ \ \ =>\ \ \ \ \ \ AV=U\sum V^{T}V \ \ \ \ \ \underset{=======>}{V^{T}V=I} \ \ \ \ AV=U\sum"  style="display:inline;vertical-align:text-top;"/>

<img src="https://latex.codecogs.com/gif.latex?AV=U\sum" title="AV=U\sum"  style="display:inline;vertical-align:text-top;"/> 写成列向量的组合形式：

<img src="https://latex.codecogs.com/gif.latex?A(v_{1},v_{2},...,v_{n})=(u_{1},u_{2},...,u_{n},...,u_{m})&space;\underset{m\times&space;n}{\begin{bmatrix}&space;\lambda_{1}&&space;&&space;&&space;&&space;0\\&space;&&space;\lambda_{2}&&space;&&space;&&space;\\&space;&&space;&&space;...&&space;&&space;\\&space;&&space;&&space;&&space;&&space;\lambda_{n}\\&space;&&space;&&space;&&space;&&space;0\\&space;&&space;&&space;&&space;&&space;...\\&space;0&space;&&space;&&space;&&space;&&space;0\\&space;\end{bmatrix}&space;}" style="display:inline;vertical-align:text-top;">

=>

<img src="https://latex.codecogs.com/gif.latex?(Av_{1},Av_{2},...,Av_{n})=(\lambda_{1}u_{1},\lambda_{1}u_{1},...,\lambda_{n}u_{n})" title="(Av_{1},Av_{2},...,Av_{n})=(\lambda_{1}u_{1},\lambda_{1}u_{1},...,\lambda_{n}u_{n})"  style="display:inline;vertical-align:text-top;"/>

=>

<img src="https://latex.codecogs.com/gif.latex?\lambda_{1}=\frac{Av_{1}}{u_{1}},&space;\&space;\&space;\&space;\lambda_{2}=\frac{Av_{2}}{u_{2}}&space;\&space;\&space;\&space;...&space;\&space;\&space;\&space;\&space;\&space;\lambda_{n}=\frac{Av_{n}}{u_{n}}" title="\lambda_{1}=\frac{Av_{1}}{u_{1}}, \ \ \ \lambda_{2}=\frac{Av_{2}}{u_{2}} \ \ \ ... \ \ \ \ \ \lambda_{n}=\frac{Av_{n}}{u_{n}}"  style="display:inline;vertical-align:text-top;"/>

另外一种推导方式——

在最开始的奇异值分解推导中，我们已经通过<img src="https://latex.codecogs.com/gif.latex?\Lambda&space;_{1}" title="\Lambda _{1}"  style="display:inline;vertical-align:text-top;"/>和<img src="https://latex.codecogs.com/gif.latex?\Lambda&space;_{2}" title="\Lambda _{2}"  style="display:inline;vertical-align:text-top;"/>来表示<img src="https://latex.codecogs.com/gif.latex?AA^{T}" title="AA^{T}"  style="display:inline;vertical-align:text-top;"/>和<img src="https://latex.codecogs.com/gif.latex?A^{T}A" title="A^{T}A"  style="display:inline;vertical-align:text-top;"/>，现在我们通过<img src="https://latex.codecogs.com/gif.latex?\sum" title="\sum"  style="display:inline;vertical-align:text-top;"/>来表示<img src="https://latex.codecogs.com/gif.latex?AA^{T}" title="AA^{T}"  style="display:inline;vertical-align:text-top;"/>和<img src="https://latex.codecogs.com/gif.latex?A^{T}A" title="A^{T}A"  style="display:inline;vertical-align:text-top;"/>。这样我们就能知道<img src="https://latex.codecogs.com/gif.latex?\Lambda&space;_{1},\Lambda&space;_{2}" title="\Lambda _{1},\Lambda _{2}"  style="display:inline;vertical-align:text-top;"/>和<img src="https://latex.codecogs.com/gif.latex?\sum" title="\sum"  style="display:inline;vertical-align:text-top;"/>的关系

<img src="https://latex.codecogs.com/gif.latex?A=U\sum&space;V^{T}&space;\&space;\&space;\&space;\&space;=&space;>&space;\&space;\&space;\&space;A^{T}&space;=V(\sum)^{T}U^{T}" title="A=U\sum V^{T} \ \ \ \ = > \ \ \ A^{T} =V(\sum)^{T}U^{T}"  style="display:inline;vertical-align:text-top;"/>

<img src="https://latex.codecogs.com/gif.latex?AA^{T}=U\sum&space;V^{T}V(\sum)^{T}U^{T}&space;=U&space;\underset{m\times&space;m}{\begin{bmatrix}&space;\lambda_{1}^{2}&&space;&&space;&&space;&&space;&&space;&0&space;\\&space;&&space;\lambda_{1}^{2}&&space;&&space;&&space;&&space;&&space;\\&space;&&space;&&space;...&&space;&&space;&&space;&&space;\\&space;&&space;&&space;&&space;\lambda_{n}^{2}&&space;&&space;&\\&space;&&space;&&space;&&space;&&space;0&&space;&\\&space;&&space;&&space;&&space;&&space;&&space;...&\\&space;0&space;&&space;&&space;&&space;&&space;&&space;&0\\&space;\end{bmatrix}}U^{T}"  style="display:inline;vertical-align:text-top;"/>

<img src="https://latex.codecogs.com/gif.latex?A^{T}A=V(\sum)^{T}U^{T}U\sum&space;V^{T}&space;=V&space;\underset{n\times&space;n}{\begin{bmatrix}&space;\lambda_{1}^{2}&&space;&&space;&&space;\\&space;&&space;\lambda_{1}^{2}&&space;&&space;\\&space;&&space;&&space;...&&space;\\&space;&&space;&&space;&&space;\lambda_{n}^{2}&space;\end{bmatrix}}V^{T}"  style="display:inline;vertical-align:text-top;"/>

BTW，上面用到了<img src="https://latex.codecogs.com/gif.latex?U^{T}U=I,\&space;V^{T}V=I" title="U^{T}U=I,\ \ \ V^{T}V=I"  style="display:inline;vertical-align:text-top;"/>的条件。

由此我们得到奇异值为特征值的开根。

### SVD分解的意义

A可以被U和V表示，而其中的奇异值<img src="https://latex.codecogs.com/gif.latex?\lambda_{1},\lambda_{2},...,\lambda_{n}" title="\lambda_{1},\lambda_{2},...,\lambda_{n}"  style="display:inline;vertical-align:text-top;"/>分别为U的列向量和V的行向量的权重：

<img src="https://latex.codecogs.com/gif.latex?\underset{m\times&space;n}{A}=\underset{m\times&space;m}{(u_{1},u_{2},...,u_{n},...,u_{m})}&space;\underset{m\times&space;n}{\begin{bmatrix}&space;\lambda_{1}&&space;&&space;&&space;0&space;\\&space;&&space;\lambda_{2}&&space;&&space;\\&space;&&space;&&space;...&&space;\\&space;&&space;&&space;&&space;\lambda_{n}\\&space;&&space;&&space;&&space;0\\&space;&&space;&&space;&&space;...\\&space;0&&space;&&space;&&space;0\\&space;\end{bmatrix}}&space;\underset{n\times&space;n}{\begin{pmatrix}&space;v^{T}_{1}\\&space;v^{T}_{2}\\&space;...\\&space;v^{T}_{n}&space;\end{pmatrix}&space;}" title="\underset{m\times n}{A}=\underset{m\times m}{(u_{1},u_{2},...,u_{n},...,u_{m})} \underset{m\times n}{\begin{bmatrix} \lambda_{1}& & & 0 \\ & \lambda_{2}& & \\ & & ...& \\ & & & \lambda_{n}\\ & & & 0\\ & & & ...\\ 0& & & 0\\ \end{bmatrix}} \underset{n\times n}{\begin{pmatrix} v^{T}_{1}\\ v^{T}_{2}\\ ...\\ v^{T}_{n} \end{pmatrix} }"  style="display:inline;vertical-align:text-top;"/>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://latex.codecogs.com/gif.latex?=(u_{1}\lambda_{1},u_{2}\lambda_{2},...,u_{n}\lambda_{n})&space;\underset{n\times&space;n}{\begin{pmatrix}&space;v^{T}_{1}\\&space;v^{T}_{2}\\&space;...\\&space;v^{T}_{n}&space;\end{pmatrix}&space;}" title="=(u_{1}\lambda_{1},u_{2}\lambda_{2},...,u_{n}\lambda_{n}) \underset{n\times n}{\begin{pmatrix} v^{T}_{1}\\ v^{T}_{2}\\ ...\\ v^{T}_{n} \end{pmatrix} }"  style="display:inline;vertical-align:text-top;"/>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://latex.codecogs.com/gif.latex?=\lambda_{1}u_{1}v^{T}_{1}&plus;\lambda_{2}u_{2}v^{T}_{2}&plus;...&plus;\lambda_{n}u_{n}v^{T}_{n}" title="=\lambda_{1}u_{1}v^{T}_{1}+\lambda_{2}u_{2}v^{T}_{2}+...+\lambda_{n}u_{n}v^{T}_{n}"  style="display:inline;vertical-align:text-top;"/>

通常，前k大的奇异值就足以占了所以奇异值90%的比重，这种情况下，我们只需要选择前k个奇异值，对应地选择U的前k个列向量，还有V的前k个行向量，就可以近似表示出矩阵A:

<img src="https://latex.codecogs.com/gif.latex?\underset{m\times&space;n}{\tilde{A}}=\underset{m\times&space;k}{(u_{1},u_{2},...,u_{k})}&space;\underset{k\times&space;k}{\begin{bmatrix}&space;\lambda_{1}&&space;&&space;&&space;0&space;\\&space;&&space;\lambda_{2}&&space;&&space;\\&space;&&space;&&space;...&&space;\\&space;&&space;&&space;&&space;\lambda_{k}\end{bmatrix}}&space;\underset{k\times&space;n}{\begin{pmatrix}&space;v^{T}_{1}\\&space;v^{T}_{2}\\&space;...\\&space;v^{T}_{k}&space;\end{pmatrix}&space;}" style="display:inline;vertical-align:text-top;">

<img src="https://github.com/DorianZi/algorithm_explained/raw/master/res/svd_cut.png" style="display:inline;vertical-align:text-top;">

这样一种降维思想在图像压缩，推荐系统里面有着非常广泛的应用。


# 参考
https://www.bilibili.com/video/av6540378/

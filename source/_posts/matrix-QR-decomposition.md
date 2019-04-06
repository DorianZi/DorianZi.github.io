---
title: QR Decomposition 矩阵的QR分解
date: 2019-03-04 01:52:44
categories: ["Technic"]
tags: ["Algorithm"]
---

矩阵的QR分解来源于gram-schmidt正交化. 先看后者

# gram-schmidt正交化
我们有<img src="https://latex.codecogs.com/gif.latex?R^{n}" style="display:inline;vertical-align:text-top;" >空间的一组基 <img src="https://latex.codecogs.com/gif.latex?\left&space;(&space;\alpha_{1},\alpha_{1},\alpha_{1},...,\alpha_{n}&space;\right&space;)" style="display:inline;vertical-align:text-top;">，并希望得到该空间的一组正交基<img src="https://latex.codecogs.com/gif.latex?\left&space;(&space;\beta_{1},\beta_{1},\beta_{1},...,\beta_{n}&space;\right&space;)" style="display:inline;vertical-align:text-top;">

1) 令<img src="https://latex.codecogs.com/gif.latex?\beta_{1}=\alpha_{1}" title="\beta_{1}=\alpha_{1}" style="display:inline;vertical-align:text-top;" style="display:inline;vertical-align:text-top;"/>，则<img src="https://latex.codecogs.com/gif.latex?\alpha_{1}" title="\alpha_{1}" style="display:inline;vertical-align:text-top;"/>被消灭

2) <img src="https://latex.codecogs.com/gif.latex?\alpha_{2}" title="\alpha_{2}" style="display:inline;vertical-align:text-top;"/>向<img src="https://latex.codecogs.com/gif.latex?\beta_{1}" title="\beta_{1}" style="display:inline;vertical-align:text-top;"/>作垂线获得<img src="https://latex.codecogs.com/gif.latex?\beta_{2}" title="\beta_{2}" style="display:inline;vertical-align:text-top;"/>，那么<img src="https://latex.codecogs.com/gif.latex?\beta_{2}" title="\beta_{2}" style="display:inline;vertical-align:text-top;"/>和<img src="https://latex.codecogs.com/gif.latex?\beta_{1}" title="\beta_{1}" style="display:inline;vertical-align:text-top;"/>正交，且可以表示<img src="https://latex.codecogs.com/gif.latex?\alpha_{2}" title="\alpha_{2}" style="display:inline;vertical-align:text-top;"/>，则<img src="https://latex.codecogs.com/gif.latex?\alpha_{2}" title="\alpha_{2}" style="display:inline;vertical-align:text-top;"/>被消灭

<img src="https://github.com/DorianZi/algorithm_explained/blob/master/res/QR_pic1.png?raw=true" style="display:inline;vertical-align:text-top;"/>

向量<img src="https://latex.codecogs.com/gif.latex?\alpha_{2}" title="\alpha_{2}" style="display:inline;vertical-align:text-top;"/>到向量<img src="https://latex.codecogs.com/gif.latex?\beta_{1}" title="\beta_{1}" style="display:inline;vertical-align:text-top;"/>的正交投影为<img src="https://latex.codecogs.com/gif.latex?\frac{(\alpha_{2},\beta_{1})}{(\beta_{1},\beta_{1})}\beta_{1}" title="\frac{(\alpha_{2},\beta_{1})}{(\beta_{1},\beta_{1})}\beta_{1}" style="display:inline;vertical-align:text-top;"/>

则<img src="https://latex.codecogs.com/gif.latex?\beta_{2}=\alpha_{2}-&space;\frac{(\alpha_{2},\beta_{1})}{(\beta_{1},\beta_{1})}\beta_{1}" title="\beta_{2}=\alpha_{2}- \frac{(\alpha_{2},\beta_{1})}{(\beta_{1},\beta_{1})}\beta_{1}" style="display:inline;vertical-align:text-top;"/>

3) <img src="https://latex.codecogs.com/gif.latex?\alpha_{3}" title="\alpha_{3}" style="display:inline;vertical-align:text-top;"/>向<img src="https://latex.codecogs.com/gif.latex?\beta_{2}" title="\beta_{2}" style="display:inline;vertical-align:text-top;"/>和<img src="https://latex.codecogs.com/gif.latex?\beta_{1}" title="\beta_{1}" style="display:inline;vertical-align:text-top;"/>表示的二维平面作垂线，获得<img src="https://latex.codecogs.com/gif.latex?\beta_{3}" title="\beta_{3}" style="display:inline;vertical-align:text-top;"/>，那么<img src="https://latex.codecogs.com/gif.latex?\beta_{1}" title="\beta_{1}" style="display:inline;vertical-align:text-top;"/>，<img src="https://latex.codecogs.com/gif.latex?\beta_{2}" title="\beta_{2}" style="display:inline;vertical-align:text-top;"/>，<img src="https://latex.codecogs.com/gif.latex?\beta_{3}" title="\beta_{3}" style="display:inline;vertical-align:text-top;"/>互相正交，且可以表示<img src="https://latex.codecogs.com/gif.latex?\alpha_{3}" title="\alpha_{3}" style="display:inline;vertical-align:text-top;"/>，则<img src="https://latex.codecogs.com/gif.latex?\alpha_{3}" title="\alpha_{3}" style="display:inline;vertical-align:text-top;"/>被消灭

<img src="https://github.com/DorianZi/algorithm_explained/blob/master/res/QR_pic2.png?raw=true" style="display:inline;vertical-align:text-top;"/>

向量<img src="https://latex.codecogs.com/gif.latex?\alpha_{3}" title="\alpha_{3}" style="display:inline;vertical-align:text-top;"/>到<img src="https://latex.codecogs.com/gif.latex?\beta_{2}" title="\beta_{2}"  style="display:inline;vertical-align:text-top;"/>和<img src="https://latex.codecogs.com/gif.latex?\beta_{1}" title="\beta_{1}"  style="display:inline;vertical-align:text-top;"/>表示的二维平面的正交投影可以分解为：到<img src="https://latex.codecogs.com/gif.latex?\beta_{2}" title="\beta_{2}"  style="display:inline;vertical-align:text-top;"/>的正交投影，以及到<img src="https://latex.codecogs.com/gif.latex?\beta_{1}" title="\beta_{1}"  style="display:inline;vertical-align:text-top;"/>的正交投影的和

所以正交投影为<img src="https://latex.codecogs.com/gif.latex?\frac{(\alpha_{3},\beta_{1})}{(\beta_{1},\beta_{1})}\beta_{1}&plus;\frac{(\alpha_{3},\beta_{2})}{(\beta_{2},\beta_{2})}\beta_{2}" title="\frac{(\alpha_{3},\beta_{1})}{(\beta_{1},\beta_{1})}\beta_{1}+\frac{(\alpha_{3},\beta_{2})}{(\beta_{2},\beta_{2})}\beta_{2}"  style="display:inline;vertical-align:text-top;"/>

则<img src="https://latex.codecogs.com/gif.latex?\beta_{3}=&space;\alpha_{3}-&space;\left&space;(\frac{(\alpha_{3},\beta_{1})}{(\beta_{1},\beta_{1})}\beta_{1}&plus;\frac{(\alpha_{3},\beta_{2})}{(\beta_{2},\beta_{2})}\beta_{2}&space;\right&space;)" title="\beta_{3}= \alpha_{3}- \left (\frac{(\alpha_{3},\beta_{1})}{(\beta_{1},\beta_{1})}\beta_{1}+\frac{(\alpha_{3},\beta_{2})}{(\beta_{2},\beta_{2})}\beta_{2} \right )"  style="display:inline;vertical-align:text-top;"/>

4) 推广到n维,<img src="https://latex.codecogs.com/gif.latex?\alpha_{n}" title="\alpha_{n}"  style="display:inline;vertical-align:text-top;"/>可以被正交向量<img src="https://latex.codecogs.com/gif.latex?\beta_{n},\beta_{n-1},...,\beta_{1}" title="\beta_{n},\beta_{n-1},...,\beta_{1}"  style="display:inline;vertical-align:text-top;"/>表示，则<img src="https://latex.codecogs.com/gif.latex?\alpha_{n}" title="\alpha_{n}"  style="display:inline;vertical-align:text-top;"/>被消灭，且：

<img src="https://latex.codecogs.com/gif.latex?\beta_{n}=\alpha_{n}-\sum_{m=1}^{n-1}\frac{(\alpha_{n},\beta_{m})}{(\beta_{m},\beta_{m})}\beta_{m}" title="\beta_{n}=\alpha_{n}-\sum_{m=1}^{n-1}\frac{(\alpha_{n},\beta_{m})}{(\beta_{m},\beta_{m})}\beta_{m}"  style="display:inline;vertical-align:text-top;"/>

以上即为gram-schmidt正交化。

# QR分解
在前面的gram-schmidt推导过程可知：

<img src="https://latex.codecogs.com/gif.latex?\alpha_{1}" title="\alpha_{1}"  style="display:inline;vertical-align:text-top;"/>可以由<img src="https://latex.codecogs.com/gif.latex?\beta_{1}" title="\beta_{1}" style="display:inline;vertical-align:text-top;"/>表示：<img src="https://latex.codecogs.com/gif.latex?\alpha_{1}=k_{11}\beta{1}" title="\alpha_{1}=k_{11}\beta{1}"  style="display:inline;vertical-align:text-top;"/>

<img src="https://latex.codecogs.com/gif.latex?\alpha_{2}" title="\alpha_{2}"  style="display:inline;vertical-align:text-top;"/>可以由正交向量<img src="https://latex.codecogs.com/gif.latex?\beta_{1}" title="\beta_{1}"  style="display:inline;vertical-align:text-top;"/>，<img src="https://latex.codecogs.com/gif.latex?\beta_{2}" title="\beta_{2}"  style="display:inline;vertical-align:text-top;"/>表示：<img src="https://latex.codecogs.com/gif.latex?\alpha_{2}=k_{12}\beta{1}&plus;k_{22}\beta_{2}" title="\alpha_{2}=k_{12}\beta{1}+k_{22}\beta_{2}"  style="display:inline;vertical-align:text-top;"/>

<img src="https://latex.codecogs.com/gif.latex?\alpha_{3}" title="\alpha_{3}"  style="display:inline;vertical-align:text-top;"/>可以由正交向量<img src="https://latex.codecogs.com/gif.latex?\beta_{1}" title="\beta_{1}"  style="display:inline;vertical-align:text-top;"/>，<img src="https://latex.codecogs.com/gif.latex?\beta_{2}" title="\beta_{2}"  style="display:inline;vertical-align:text-top;"/>，<img src="https://latex.codecogs.com/gif.latex?\beta_{3}" title="\beta_{3}"  style="display:inline;vertical-align:text-top;"/>表示：<img src="https://latex.codecogs.com/gif.latex?\alpha_{3}=k_{13}\beta{1}&plus;k_{23}\beta_{2}&plus;k_{33}\beta_{3}" title="\alpha_{3}=k_{13}\beta{1}+k_{23}\beta_{2}+k_{33}\beta_{3}"  style="display:inline;vertical-align:text-top;"/>

......

<img src="https://latex.codecogs.com/gif.latex?\alpha_{n}" title="\alpha_{n}"  style="display:inline;vertical-align:text-top;"/>可以由正交向量<img src="https://latex.codecogs.com/gif.latex?\left&space;(&space;\beta_{1},\beta_{2},\beta_{3},...,\beta_{n}&space;\right&space;)" title="\left ( \beta_{1},\beta_{2},\beta_{3},...,\beta_{n} \right )"  style="display:inline;vertical-align:text-top;"/>表示：<img src="https://latex.codecogs.com/gif.latex?\alpha_{n}=k_{1n}\beta{1}&plus;k_{2n}\beta_{2}&plus;k_{3n}\beta_{3}&plus;...&plus;k_{nn}\beta_{n}" title="\alpha_{n}=k_{1n}\beta{1}+k_{2n}\beta_{2}+k_{3n}\beta_{3}+...+k_{nn}\beta_{n}"  style="display:inline;vertical-align:text-top;"/>

写成矩阵乘法格式A=QR：

<img src="https://latex.codecogs.com/gif.latex?(&space;\alpha_{1},\alpha_{2},\alpha_{3},...,\alpha_{n}&space;)=&space;(\beta_{1},\beta_{2},\beta_{3},...,\beta_{n})&space;\begin{bmatrix}&space;k_{11}&space;&&space;k_{12}&space;&&space;k_{13}&space;&&space;...&space;&&space;k_{1n}\\&space;0&space;&&space;k_{22}&space;&&space;k_{23}&space;&&space;...&space;&&space;k_{2n}\\&space;0&space;&&space;0&space;&&space;k_{33}&space;&&space;...&space;&&space;k_{3n}\\&space;...&space;&&space;...&space;&&space;...&space;&&space;...&space;&&space;...\\&space;0&space;&&space;0&space;&&space;0&space;&&space;...&space;&&space;k_{nn}&space;\end{bmatrix}\right" title="( \alpha_{1},\alpha_{2},\alpha_{3},...,\alpha_{n} )= (\beta_{1},\beta_{2},\beta_{3},...,\beta_{n}) \begin{bmatrix} k_{11} & k_{12} & k_{13} & ... & k_{1n}\\ 0 & k_{22} & k_{23} & ... & k_{2n}\\ 0 & 0 & k_{33} & ... & k_{3n}\\ ... & ... & ... & ... & ...\\ 0 & 0 & 0 & ... & k_{nn} \end{bmatrix}\right"  style="display:inline;vertical-align:text-top;"/>

以上即为A=QR分解。


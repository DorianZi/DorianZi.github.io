---
title: Image Compression by SVD
date: 2019-03-19 09:53:44
tags:
---

# 基于SVD的图像压缩实践

在[之前的介绍](https://github.com/DorianZi/algorithm_explained/blob/master/matrix_SVD_decomposition.md)里，我们讲过了SVD分解的基本原理。一个mxn的矩阵A可以分解为三个矩阵相乘：

A可以被U和V表示，而其中的奇异值<img src="https://latex.codecogs.com/gif.latex?\lambda_{1},\lambda_{2},...,\lambda_{n}" title="\lambda_{1},\lambda_{2},...,\lambda_{n}"  style="display:inline;vertical-align:text-top;"/>分别为U的列向量和V的行向量的权重：

<img src="https://latex.codecogs.com/gif.latex?\underset{m\times&space;n}{A}=\lambda_{1}u_{1}v^{T}_{1}&plus;\lambda_{2}u_{2}v^{T}_{2}&plus;...&plus;\lambda_{n}u_{n}v^{T}_{n}" title="=\lambda_{1}u_{1}v^{T}_{1}+\lambda_{2}u_{2}v^{T}_{2}+...+\lambda_{n}u_{n}v^{T}_{n}"  style="display:inline;vertical-align:text-top;"/>

通常，前k大的奇异值就足以占了所以奇异值90%的比重，这种情况下，我们只需要选择前k个奇异值，对应地选择U的前k个列向量，还有V的前k个行向量，就可以近似表示出矩阵A:

<img src="https://latex.codecogs.com/gif.latex?\underset{m\times&space;n}{\tilde{A}}=\underset{m\times&space;k}{(u_{1},u_{2},...,u_{k})}&space;\underset{k\times&space;k}{\begin{bmatrix}&space;\lambda_{1}&&space;&&space;&&space;0&space;\\&space;&&space;\lambda_{2}&&space;&&space;\\&space;&&space;&&space;...&&space;\\&space;&&space;&&space;&&space;\lambda_{k}\end{bmatrix}}&space;\underset{k\times&space;n}{\begin{pmatrix}&space;v^{T}_{1}\\&space;v^{T}_{2}\\&space;...\\&space;v^{T}_{k}&space;\end{pmatrix}&space;}" style="display:inline;vertical-align:text-top;">

在图像处理中，我们把一张图片的每个像素点作为矩阵中的元素，则我们得到一个mxn的矩阵，利用SVD分解，取前k个奇异值，就能实现图像的压缩：

<img src="https://github.com/DorianZi/algorithm_explained/raw/master/res/svd_cut.png" style="display:inline;vertical-align:text-top;">

同时我也可以称作其为图像降噪，因为可以认为奇异值非常小的那些项是噪声。
Anyway, 直接上代码

## 完整代码
```
import matplotlib.pyplot as plt
from optparse import OptionParser
import numpy as np
import sys

# Get image path
def getOptions():
	parser = OptionParser()
	parser.add_option("-p", "--picture", dest="picture", action="store", help="specify the input picture path")
	parser.add_option("-c", "--count", dest="count", action="store", help="specify the required singular value count")
	options, _ = parser.parse_args()
	if not options.picture:
		sys.exit("-p/--picture is required!")
	return options


def compress(K,U,Sigma,VT,verbose=False):
	if verbose:
		print("Selected singular value count: {}".format(K))
		print("Before compression: {} {} {}".format(U.shape, Sigma.shape, VT.shape))
	U_K = U[:, :K]
	Sigma_K = np.eye(K)*Sigma[:K]   # Because Sigma matrix is a vector instead of a matrix, need to convert to matrix
	VT_K = VT[:K, :]
	if verbose:
		print("After  compression: {} {} {}".format(U_K.shape, Sigma_K.shape, VT_K.shape))
	return np.matmul(np.matmul(U_K,Sigma_K), VT_K)



if __name__ == "__main__":
	options = getOptions()

	# Read image
	pMatrix= np.array(plt.imread(options.picture))
	print("Original image shape: {}".format(pMatrix.shape))

	# Do SVD decomposition for each channel of RGB
	R, G, B = pMatrix[:,:,0], pMatrix[:,:,1], pMatrix[:,:,2]
	U_R,Sigma_R,VT_R = np.linalg.svd(R)
	U_G,Sigma_G,VT_G = np.linalg.svd(G)
	U_B,Sigma_B,VT_B = np.linalg.svd(B)

	# compress by top K singular values
	K = int(options.count) if options.count else 100
	R_new = compress(K,U_R,Sigma_R,VT_R,verbose=True)
	G_new = compress(K,U_G,Sigma_G,VT_G)
	B_new = compress(K,U_B,Sigma_B,VT_B)
	pMatrix_new = np.stack((R_new,G_new,B_new),2)  # Compose R,G,B channels back to complete image
	print("Compresses image shape: {}".format(pMatrix_new.shape))

	# show image after compress
	plt.imshow(pMatrix_new)
	plt.show()
	sys.exit()



```

## 运行结果
```
 $ python imageCompress.py -p pic_1.png -c 1000
      Original image shape: (1440L, 1080L, 4L)
      Selected singular value count: 1000
      Before compression: (1440L, 1440L) (1080L,) (1080L, 1080L)
      After  compression: (1440L, 1000L) (1000L, 1000L) (1000L, 1080L)
      Compresses image shape: (1440L, 1080L, 3L)
```
当K=10，即选取10个奇异值进行压缩：

<img src="https://github.com/DorianZi/algorithm_explained/blob/master/res/SVD_k_10.png?raw=true" style="display:inline;vertical-align:text-top;">

当K=30

<img src="https://github.com/DorianZi/algorithm_explained/blob/master/res/SVD_k_30.png?raw=true" style="display:inline;vertical-align:text-top;">

当K=50

<img src="https://github.com/DorianZi/algorithm_explained/blob/master/res/SVD_k_50.png?raw=true" style="display:inline;vertical-align:text-top;">

当K=100

<img src="https://github.com/DorianZi/algorithm_explained/blob/master/res/SVD_k_100.png?raw=true" style="display:inline;vertical-align:text-top;">

当K=500

<img src="https://github.com/DorianZi/algorithm_explained/blob/master/res/SVD_k_500.png?raw=true" style="display:inline;vertical-align:text-top;">

可见，当使用前100个奇异值（约占全部奇异值数量的10%），就可以比较好地表示原图像了

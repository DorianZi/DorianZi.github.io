---
title: Face Recognition by SVM or PCA 基于SVM或PCA的人脸识别
date: 2019-05-23 10:21:59
tags: ["SVM","PCA","OpenCV"]
categories: Technic
---

本文将**分别**讲述基于SVM和PCA的人脸识别

# SVM人脸识别理论

在之前的[文章](https://dorianzi.github.io/2019/04/01/SVM/)中，我们讲过SVM的原理：利用超平面进行二分类。我们来看看它是怎么作用到人脸识别上来的。

人脸作为一张h\*w的二维图像(去掉颜色通道的灰度图)，可以看做一个h\*w维的（含有h\*w个坐标值）的数据。在SVM里，寻找到一个h\*w-1维的超平面可以对这样的人脸进行二分类。假设我们有两个人的多张人脸数据，就可以训练出一个模型（找到这样一个超平面），能够最好地区分两种人脸。

在实际操作中，我们不会只满足于二分类的人脸识别。事实上SVM通过组合可以构建多分类的分类器，通常的实现方法是：1）对其中一种脸和**其它**的脸构建二分类模型，轮流构建多个即可。2）对任何两种脸构建二分类模型，遍历所有组合即可。

本文中使用了sklearn中的SVM的api，就是有多分类功能的。

# PCA人脸识别理论

PCA原来在之前的[文章](https://dorianzi.github.io/2019/03/19/PCA/)也有讲述：通过在数据们的信息量最大的方向上进行降维投影，获得数据的在新空间的表示。

谈到人脸识别或者任何形式的图片相似度匹配，假设我们不知道PCA或SVM这种成熟的算法原理，而需要这个问题。容易想到：把h\*w大小的图片看做一个h\*w维数据，这样在h\*w维空间里，一张图片只是一个点。考察两张图片（两个点）在h\*w维空间的距离，距离越近则相似度越高。

事实上PCA最终的解决方案跟上述方法是一致的！只是PCA的精髓在于，提供了一种降维变换。在降维变换到新的空间之后，用简化的方式表示原来的图片。然后再考虑在新空间中两张图片的距离来确定相似度。

具体到PCA在人脸识别的应用，就是：先找到训练人脸数据的投影空间（特征脸空间），然后将训练的人脸们投影到该空间。在识别时，将待识别的脸也投影到该空间，然后逐一比较它与训练的人脸的距离。距离最近则为识别出的对象。

再具体一点，在PCA人脸识别的工程实践中，我们使用的方法叫做**Eigenfaces**，即特征脸方法。它不是直接将训练人脸投影到特征脸空间，而是先进行中心化，即求所有训练数据的平均值（称作“平均脸”），求得所有训练人脸到该平均值的差值作为真正的训练数据，再进行投影。在识别时，也会先将待识别的人脸减去上述平均脸，再投影，再计算距离。

下面是特征脸。每一张脸其实就是一个特征向量，他们作为该空间的一组基存在。

![](/uploads/face_detect_1.png)

# 实现原理和代码

完整代码[见此](https://github.com/DorianZi/projects/tree/master/face-detect)
在代码上，SVM和PCA没有什么区别，因为只是OpenCV的接口不同。除此之外的部分是一致的。

在代码实现上，首先通过传参来决定：人脸采集 or 人脸识别。 人脸采集通过OpenCV调用计算机的摄像头进行图像拍照，然后抠出人脸进行保存。人脸识别上，同样通过OpenCV调用计算机摄像头进行拍照，此外还可以通过输入图像文件进行识别。值得一提的是，通过循环拍照识别，可以将识别过程动态化为视频识别。

下面代码，我们按照功能而非按照工作流程进行讲解

## 图片读取
```
import cv2
from sklearn.svm import SVC
import numpy as np

# 创建摄像头操作对象
camera = cv2.VideoCapture(0)

# 通过摄像头读取一张BGR图像，image的格式为numpy.ndarray，image.shape为(480, 640, 3)
success, image = camera.read()

# 如果是通过问文件读取则为image = cv2.imread(PICTURE_PATH)

# 将图像转换为灰度图，shape为(480, 640)
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

```

## 人脸提取

OpenCV自带了人脸提取的Haar特征分类器，其参数以xml格式保存，只要调出相应文件即可使用
```
# 通过CascadeClassifier级联分类器读取模型参数，获得模型对象
face_cascade = cv2.CascadeClassifier('<PATH>/haarcascade_frontalface_alt2.xml')

# 通过detectMultiScale提取gray_img里的每个人脸的方形坐标
# 输出faces为numpy.ndarray格式的x,y,w,h数据，比如[[238 109 217 217]，[121 19 215 215]]，代表两个人脸
# 第一个人脸框左上顶点为(238,109)，宽和高分别为217和217；第二个人脸框左上顶点为(121,19)，宽和高分别为215和215
faces = face_cascade.detectMultiScale(gray_img, 1.1, 5)

# 打印看看
for x,y,w,h in faces:
    cv2.imshow('image', gray_img[y:y+h,x:x+w])
    cv2.waitKey(0)
```

经过实际测试，OpenCV自带的模型精度并不高，当图片中的人脸有倾斜或者侧转的时候，很容易提取不出人脸。
为了更高的精度，我们使用著名的经过caffe训练好的人脸提取器。OpenCV的dnn库有相关的接口可以直接读取caffe模型文件：

```
# 实例化一个caffe模型
caffe_detector = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

# 将(480, 640, 3)的image转换为(1, 3, 300, 300)，便于后续计算处理
imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),\
                                  (104.0, 177.0, 123.0), swapRB=False, crop=False)

# 输入图片
caffe_detector.setInput(imageBlob)

# 进行前向计算，获取预测输出。该输出detections的shape为(1, 1, 89, 7)
# 这意味着一共抓了89个框，每个框有7个数值信息
detections = caffe_detector.forward()

# 获取图片高和宽
(h, w) = image.shape[:2]
faceBoxList = []

# 遍历这89个框，筛选出所有人脸框
for i in range(detections.shape[2]):
    # 7个数值信息的第0,1位这里不用，第2位是人脸置信度，第3~6位是提取框的顶点坐标和宽高归一化值
    confidence = detections[0, 0, i, 2]

    # 当置信度大于50%才认为该提取框为人脸
    if confidence >  0.5:
        # 获取图片数据下的提取框坐标的绝对数值
        faceBox = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype("int")
        faceBoxList.append(faceBox)

# 如果需要将提取出来的框保存为图片，则执行下面代码
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 只保存灰度图
for i, (startX, startY, endX, endY) in enumerate(faceBoxList):
    cv2.imwrite(os.path.join(STORE_PATH,"picture_name_{}.jpg".format(i), cv2.resize(gray_img[startY:endY, startX:endX],(300,300)))

```

## 训练

假设我们已经获取了很多张人脸图片（灰度图）及其label，将这些图片和label读取到numpy.ndarray变量train_X, train_Y中。 其中train_X的shape为(N,300,300)，train_Y的shape为(N,)
同时我们待预测的原始图片为image,其灰度图为gray_img，其人脸所在框的两个顶点为(startX, startY), (endX, endY)

### SVM
```
from sklearn.svm import SVC

# 使用线性核函数
model = SVC(C=1.0, kernel="linear", probability=True)

# 将训练数据扁平化
model.fit(train_X.reshape(train_X.shape[0],-1), train_Y)

# 将待识别人脸图像数据标准化为(300,300)再扁平化
target = cv2.resize(gray_img[startY:endY, startX:endX],(300,300)).reshape(1,-1)

# 预测结果。返回为label
pred = model.predict(target)

# 预测label所对应的置信度
pred_proba = model.predict_proba(target)[0][pred[0]-1]

# 用绿色框将人脸标出来
cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

# 在框上面写上文字：人名和置信度百分比。 其中pred_dict是label到name的mapping
cv2.putText(image,pred_dict[pred[0]] + " %.2f%%" % (100*float(pred_proba)) ,(startX,startY-20),cv2.FONT_HERSHEY_SIMPLEX,1,255,2)

# 显示图片
cv2.imshow('image', image)

# 无限等待，直到回车或者关闭图片
cv2.waitKey(0)
```

### PCA
```
# 实例化特征脸识别器对象
model = cv2.face.EigenFaceRecognizer_create()

# train_X无需扁平化
model.train(train_X, train_Y)

# 标准化为(300,300)
target = cv2.resize(gray_img[startY:endY, startX:endX],(300,300))

# 预测结果。返回为（label，系数），其中系数是一个跟算法有关的数值，我们需要的是label
pred = model.predict(target)

# 用绿色框将人脸标出来
cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

# 在框上面写上文字。 其中pred_dict是label到name的mapping
cv2.putText(image,pred_dict[pred[0]],(startX,startY-20),cv2.FONT_HERSHEY_SIMPLEX,1,255,2)

# 显示图片
cv2.imshow('image', image)

# 无限等待，直到回车或者关闭图片
cv2.waitKey(0)
```

## 效果展示

![](/uploads/face_detect_2.png)


# 参考
https://docs.opencv.org/master/db/d7c/group__face.html
https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/

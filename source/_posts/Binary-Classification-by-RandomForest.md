---
title: Binary Classification by RandomForest 基于随机森林的二分类问题实践
date: 2019-05-11 08:15:37
tags: ["Machine Learning", "sklearn"]
categories: Technic
---

# 问题的提出

问题来自于Kaggle的一个[Titanic](https://www.kaggle.com/c/titanic)的竞赛项目：
给出泰坦尼克号上的乘客的特征（舱位，年龄，性别等），预测Ta是否被选中上援救船

模型训练和预测代码在[此](https://github.com/DorianZi/kaggle/blob/master/titanic/titanic.py)（不包含本文中的分析绘图部分代码）

# 数据集的获取

数据集在[这里](https://www.kaggle.com/c/titanic/data)可以下载

我们来看看训练数据(测试数据只是少了label所在的列)

![](/uploads/Binary_classification_randomForest_1.png)

需要解释的列：
- Survived 列为是否被解救的label
- pclass 舱位等级
- SibSp 同船配偶以及兄弟姐妹的人数
- Parch 同船父母或者子女的人数
- Fare 船票价格
- Cabin 舱位
- Embarked 登船港口

## 代码
```
import pandas as pd

train_df = pd.read_csv("train.csv")
train_label_data = train_df["Survived"].values
train_data_df = train_df.drop(["Survived"],axis=1)
test_df = pd.read_csv("test.csv")

# 将训练和测试数据拼接，以便于统一做特征工程
train_test_df = pd.concat([train_data_df,test_df],axis=0)

# 拿到测试集的PassengerId以便于后面的预测格式输出
test_df_ids = test_df['PassengerId'].values

```


# 特征工程

为什么要进行特征工程？ 因为特征众多，干扰也多，并不是所有特征都是需要保留的；需要保留的特征也并非直接可用的（比如需要进行数值转换，需要字符串截取）；并不是所有特征都是有数据的（需要进行数据补全，或者直接舍弃该特征）

首先对训练数据有个大致的轮廓：
```
train_test_df.info()

############################输出如下#################################
  from numpy.core.umath_tests import inner1d
<class 'pandas.core.frame.DataFrame'>
Int64Index: 1309 entries, 0 to 417
Data columns (total 11 columns):
PassengerId    1309 non-null int64
Pclass         1309 non-null int64
Name           1309 non-null object
Sex            1309 non-null object
Age            1046 non-null float64
SibSp          1309 non-null int64
Parch          1309 non-null int64
Ticket         1309 non-null object
Fare           1308 non-null float64
Cabin          295 non-null object
Embarked       1307 non-null object
dtypes: float64(2), int64(4), object(5)
memory usage: 122.7+ KB
None
```
可以看到Age，Fare，Cabin，Embarked是有缺失值的

## Pclass

Pclass在数据集里面只有1,2,3三个值，且没有缺失值，可以直接用来训练。查看它对生还率的影响
```
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

# 查看对生还率的影响，只能通过训练数据，因为测试数据没有label
pd.crosstab(train_df['Pclass'],train_df['Survived']).plot(kind='bar')
plt.show()
```

![](/uploads/Binary_classification_randomForest_11.png)

可见，等级越低，生还率越低。

为了理解pd.crosstab做了什么，查看一下它的返回数据：

```
print pd.crosstab(train_df['Pclass'],train_df['Survived'])
####打印如下####
# Survived    0    1
# Pclass            
# 1          80  136
# 2          97   87
# 3         372  119
```

## Name

名字中的Mr.  Mrs. Miss Rev. Dr.可能是身份的象征，它影响是否被解救的结果吗？我们可以通过数据可视化来观察:

```
def findTittle(name):
    title = name.split(", ")[1].split(".")[0]
    return title if title else None

# 通过apply和lambda来进行数据替换的mapping
train_test_df["Name"] = train_test_df["Name"].apply(lambda x: findTittle(x) if findTittle(x) else x)

# 为绘制生还率图做准备
train_df["Name"] = train_df["Name"].apply(lambda x: findTittle(x) if findTittle(x) else x)

# 找到集合
title_list =  train_df["Name"].unique()
survived_num = list()
not_survived_num = list()

for t in title_list:
    # train_df["Name"] == t 返回的是Series格式的数值为布尔值的数据，loc可以接受Series作为参数，并返回其中为True的值
    bar_name = train_df.loc[train_df["Name"] == t]

    # Series的函数sum()返回数值为True的个数
    survived_num.append((bar_name["Survived"] == 1).sum())
    not_survived_num.append((bar_name["Survived"] == 0).sum())

# 这里我们不使用pd.crosstab这么封装性高的来构建生还率条状图。 width=0.3设置每个条状图宽度为0.3
plt.bar(np.arange(len(title_list)),survived_num,label='survived', width=0.3)

# 构建非生还率条状图，每个条状图宽度为0.3，每隔位置平移0.3，这样就能与生还率条状图并排
plt.bar(np.arange(len(title_list))+0.3,not_survived_num,label='non-survived',width=0.3)

# 将位置映射为字符来显示
plt.xticks([i for i in np.arange(len(title_list))], title_list)

# 显示图例
plt.legend()
plt.show()
```
输出如下：

![](/uploads/Binary_classification_randomForest_2.png)

从图上看Mr生还率最低，而Mrs和Miss的生还率更高。所以title这个特征是可以影响到结果的。


## Sex

第一反应会好奇是否女性会生还率更高？我们还是通过数据来观察：

```
pd.crosstab(train_df['Sex'],train_df['Survived']).plot(kind='bar')
plt.show()
```
![](/uploads/Binary_classification_randomForest_3.png)

的确女性生还率更高，所以性别对结果有影响。

## Age

首先年龄有缺失值，需要进行缺失值处理；其次年龄跨度很大，数值众多，为了简化计算同时减少过拟合，需要对年龄进行分段。

首先看缺失值补全。

缺失值怎么补全？根据实际需要可以补零，显然这里不适用；可以补充平均值，作为年龄的话也不是很合理；可以联合其他跟年龄有关的特征，比如上面讲的title来分别对不同的title求平均，这里我们选用这个方法。

```
# 根据Name来分类，返回DataFrameGroupBy格式数据，是一种中间格式，不能打印
name_group = train_test_df.groupby("Name")

# 选中列，返回SeriesGroupBy，仍然是中间数据
age_by_name = name_group['Age']

# 进行计算，返回Series格式数据
median_age = age_by_name.median()
print median_age

########打印如下#######
# Name
# Capt            70.0
# Col             54.5
# Don             40.0
# Dona            39.0
# Dr              49.0
# Jonkheer        38.0
# Lady            48.0
# Major           48.5
# Master           4.0
# Miss            22.0
# Mlle            24.0
# Mme             24.0
# Mr              29.0
# Mrs             35.5
# Ms              28.0
# Rev             41.5
# Sir             49.0
# the Countess    33.0
# Name: Age, dtype: float64


print train_test_df["Age"]
########打印如下#######
# 0      22.0
# 1      38.0
# 2      26.0
# 3      35.0
# 4      35.0
# 5       NaN
# 6      54.0
# 7       2.0
# ...    ...

# 使用SeriesGroupBy数据通过transform填充空值
train_test_df["Age"] = age_by_name.transform(lambda x: x.fillna(x.median()))   # 这里lambda的x取值范围是每一个group里的数据
print train_df["Age"]

########打印如下#######
# 0      22.0
# 1      38.0
# 2      26.0
# 3      35.0
# 4      35.0
# 5      29.0
# 6      54.0
# 7       2.0
# ...     ...
```
groupBy数据 在进行任何操作比如mean()之前只能是中间数据。

groupby的操作图如下：

![](/uploads/Binary_classification_randomForest_4.png)

接下来进行年龄分段

```
# 查看最大最小年龄，以确定分段的首尾
print train_test_df["Age"].describe()

######打印如下#######
# count    1309.000000
# mean       29.432521
# std        13.163767
# min         0.170000
# 25%        22.000000
# 50%        29.000000
# 75%        35.500000
# max        80.000000
# Name: Age, dtype: float64


# 将年龄分段为（0,10],(10,20],...
train_test_df["Age"] =  pd.cut(train_df["Age"],[0,10,20,30,40,50,60,70,80,90])
print train_test_df["Age"]

######打印如下#######
# 0      (20, 30]
# 1      (30, 40]
# 2      (20, 30]
# 3      (30, 40]
# 4      (30, 40]
# 5      (20, 30]
# 6      (50, 60]
# ...
# 415    (30, 40]
# 416    (20, 30]
# 417     (0, 10]
# Name: Age, Length: 1309, dtype: category
# Categories (9, interval[int64]): [(0, 10] < (10, 20] < (20, 30] < (30, 40] ... (50, 60] < (60, 70] < (70, 80] < (80, 90]]


# 绘制生还图
train_df["Age"] = train_df.groupby("Name")['Age'].transform(lambda x: x.fillna(x.median()))
train_df["Age"] =  pd.cut(train_df["Age"],[0,10,20,30,40,50,60,70,80,90])
pd.crosstab(train_df['Age'], train_df['Survived']).plot(kind = 'bar')
plt.show()

```
绘制图如下：

![](/uploads/Binary_classification_randomForest_5.png)

可见儿童的生还率更高。

## SibSp

SibSp 没有缺失值，且已经是可使用的特征编码，来看看它对结果的影响

```
sibSp_Survived = pd.crosstab(train_df['SibSp'], train_df['Survived'])
sibSp_Survived.plot(kind = 'bar')
plt.show()
```
![](/uploads/Binary_classification_randomForest_6.png)

可见当SibSp为1,2时，生还率比较高。

## Parch

Parch 没有缺失值，且已经是可使用的特征编码

```
sibSp_Survived = pd.crosstab(train_df['Parch'], train_df['Survived'])
sibSp_Survived.plot(kind = 'bar')
plt.show()
```
![](/uploads/Binary_classification_randomForest_7.png)

## Ticket

```
print train_test_df["Ticket"].describe()

######打印如下#######
# count         1309
# unique         929
# top       CA. 2343
# freq            11
# Name: Ticket, dtype: object

```
因为只有929张票号，意味着有人共用船票。共用船票可能是一个影响结果的新的特征，我们区分来看看：

```
# 按照船票分组，as_index = Fals意味着船票也将作为一列，所以返回DataFrame而非Series
ticket_group_df = train_test_df.groupby("Ticket",as_index = False)['PassengerId'].count()
print ticket_group_df
##############打印如下################
#                 Ticket  PassengerId
# 0               110152            3
# 1               110413            3
# 2               110465            2
# 3               110564            1
# 4               110813            1
# 5               111240            1
#...

# 将ticket_group_df中满足非共享票的行抽取出来
one_ticket_df = ticket_group_df[ticket_group_df['PassengerId']==1]

# 获得非共享票票号的列
one_ticket_series = one_ticket_df['Ticket']

# 创造新的列，来表示是否为共享票，当满足train_test_df['Ticket'].isin(one_ticket_series)，值为0，否则为1
train_test_df['Ticket'] = np.where(train_test_df['Ticket'].isin(one_ticket_series),0, 1)



# 绘制生还图
train_ticket_group_df = train_df.groupby("Ticket",as_index = False)['PassengerId'].count()
train_one_ticket_df = train_ticket_group_df[train_ticket_group_df['PassengerId']==1]
train_one_ticket_series = train_one_ticket_df['Ticket']
train_df['Ticket'] = np.where(train_df['Ticket'].isin(train_one_ticket_series),0, 1)
GroupTicket_Survived = pd.crosstab(train_df['Ticket'],train_df['Survived'])
GroupTicket_Survived.plot(kind='bar')
plt.show()

```
![](/uploads/Binary_classification_randomForest_8.png)

可以发现，共享票（有同伴）的生存率更高。

## Fare

Fare是连续数值，有一个缺失值，需要补零之后（补零是因为空缺的Fare认为是无票或0价格票，再说空缺值只有1个），进行分段

```
# 空缺值填补为0
train_test_df['Fare'] = train_test_df['Fare'].fillna(0)

print train_test_df['Fare'].describe()
#######打印如下########
# count    1308.000000
# mean       33.270043
# std        51.747063
# min         0.000000
# 25%         7.895800
# 50%        14.454200
# 75%        31.275000
# max       512.329200
# Name: Fare, dtype: float64

# 因为取值为0~513，所以选择每50为一段分，直到550
train_test_df['Fare'] = pd.cut(train_test_df['Fare'],[-1,50,100,150,200,250,300,350,400,450,500,550])

# 绘制生还图
train_df['Fare'] = train_df['Fare'].fillna(0)
train_df['Fare'] = pd.cut(train_df['Fare'],[-1,50,100,150,200,250,300,350,400,450,500,550])
pd.crosstab(train_df['Fare'],train_df['Survived']).plot(kind='bar')
plt.show()
```

![](/uploads/Binary_classification_randomForest_9.png)

可见票价贵的，生还率相对更高

## Cabin

Cabin含有大量的缺失值。我们是否要删掉这个特征呢？其实可以保留，因为确实的值是有信息的：可能他们属于没有舱位的散座乘客。那么我们可以把散座设置为“No”。同时我们猜想有舱位和无舱位应该是会有显著的差别的，所以我们可以确定区别之后，将特征简化为有无舱位两种值

```
train_test_df['Cabin'] = train_test_df['Cabin'].fillna("No")
train_test_df['Cabin'] = np.where(train_test_df['Cabin']=='No',0,1)

# 绘制生还图
train_df['Cabin'] = train_df['Cabin'].fillna("No")
train_df['Cabin'] = np.where(train_df['Cabin']=='No',0,1)
pd.crosstab(train_df['Cabin'],train_df['Survived']).plot(kind='bar')
plt.show()
```

![](/uploads/Binary_classification_randomForest_12.png)

可见的确无舱位的生还率大大低于有舱位的。我们接受这次特征转换

## Embarked

含有两个缺失值，可以通过众数来填补

```
# Series调用mode函数，会返回Series格式的众数表格，而DataFrame会返回DataFrame
train_test_df['Embarked'] = train_test_df['Embarked'].fillna(train_test_df['Embarked'].mode()[0])

# 绘制生还图
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
pd.crosstab(train_df['Embarked'],train_df['Survived']).plot(kind='bar')
plt.show()

```

![](/uploads/Binary_classification_randomForest_10.png)

C港登船的生存率明显更大

## 创造新特征Family

我们发现SibSp和Parch两个意义一致，都是表示家庭成员，他们的图像也很相近。这个两个特征相关性非常高的，可以把它们合并为一个特征Family，用来表示家庭成员的个数
Family = SibSp + Parch + 1

```
#创造新的特征
train_test_df['Family'] = train_test_df['SibSp']+train_test_df['Parch']+1

# 绘制生还图
train_df['Family'] = train_df['SibSp']+train_df['Parch']+1
pd.crosstab(train_df['Family'],train_df['Survived']).plot(kind='bar')
plt.show()
```

![](/uploads/Binary_classification_randomForest_13.png)

可见Family跟SibSp，Parch的图像相似，当独身一人时生还率比较低，当有一两个兄弟或父母在船上时，生还率比较高。

## 特征删除

到目前为止我们有下面的特征，且根据对结果是否有影响以及特征的合并，我们考虑是否删除该特征：
PassengerId 删除
Pclass 保留
Name 提取身份，保留
Sex 保留
Age 分段，保留
SibSp 删除
Parch 删除
Family 新建，保留
Ticket 转换为是否共享票，保留
Fare 分段，保留
Cabin 转换为是否有舱位，保留
Embarked 保留

```
train_test_df = train_test_df.drop(["PassengerId","SibSp","Parch"],axis=1)

print train_test_df.columns

####打印如下####
# Index([u'Pclass', u'Name', u'Sex', u'Age', u'Ticket', u'Fare', u'Cabin', u'Embarked', u'Family'], dtype='object')
```

## 特征量化

定性的特征需要转换为数值特征，以便进行特征分裂

```
# 先获得类似{"Mr":0,"Miss":1}的字典，然后通过map来转换
train_test_df['Name'] = train_test_df['Name'].map({name:i for i,name in enumerate(train_df["Name"].unique())})

train_testdf['Sex'] = train_df['Sex'].map({name:i for i,name in enumerate(train_df["Sex"].unique())})

train_test_df['Age'] = train_test_df['Age'].map({name:i for i,name in enumerate(train_test_df["Age"].unique())})

train_test_df['Fare'] = train_test_df['Fare'].map({name:i for i,name in enumerate(train_test_df["Fare"].unique())})

train_df['Embarked'] = train_df['Embarked'].map({name:i for i,name in enumerate(train_df["Embarked"].unique())})
```

# 训练和预测

特征工程完毕之后，再次把训练和测试数据分离
```
train_data_df = train_test_df[:len(train_data_df)]
test_df = train_test_df[len(train_data_df):]
```

进行训练和预测
```
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 从训练集里切割0.2为验证集，剩下0.8为训练集
X_train, X_test, Y_train, Y_test = train_test_split(train_data_df.values, train_label_data, test_size=0.2)

# 使用随机森林
rft = RandomForestClassifier(random_state=30)
rft.fit(X_train,Y_train)

# 用验证集进行评分
print rft.score(X_test,Y_test)

# 评分结果不固定，因为每次执行训练得到的模型不固定
###打印如下####
# 0.8324022346368715

# 预测
predict = rft.predict(test_df.values)
print predict

####打印如下####
# [0 0 0 0 0 0 1 0 1 0 0 1 1 0 1 1 0 0 0 1 0 1 1 0 1 0 1 0 0 0 0 0 1 1 1 0 0 0 0 1 0 1 0 1 1 0 0 0 1 1 0 0 1 1 0 0 0 0 0 1 0 0 0 1 1 1 1 0 1 1 1 0 0 1 1 0 0 1 0 1 1 0 1 0 0 0 1 0 1 1 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 0 1 0 1 1 1 1 0 0 1 0 1 1 0 1 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 1 0 1 0 0 1 0 0 1 0 0 0 1 1 1 0 0 0 0 0 1 0 0 0 0 0 0 1 1 1 1 1 0 1 1 0 1 0 1 0 0 0 0 1 1 0 1 0 1 0 0 0 1 1 1 1 0 0 0 0 1 0 0 0 0 1 1 0 1 0 1 0 1 0 1 0 1 1 0 1 0 0 0 1 0 0 1 0 0 0 1 1 1 1 0 0 1 0 1 0 1 1 1 0 0 0 0 0 0 0 1 0 0 0 1 1 0 0 1 0 0 0 0 0 1 1 0 1 0 0 0 0 0 1 1 1 1 0 0 0 0 0 1 1 0 1 0 0 1 0 1 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 1 1 0 1 0 0 0 1 1 0 1 0 1 0 0 0 1 0 0 0 1 1 0 0 1 0 1 1 0 0 0 1 0 1 0 0 0 0 1 1 0 1 0 0 1 1 0 0 1 0 0 1 1 0 0 1 0 0 0 1 1 0 1 0 0 0 0 0 1 1 0 0 1 0 1 0 0 1 0 1 0 0 0 0 0 1 1 1 1 0 0 1 0 0 1]

```

以上。
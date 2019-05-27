---
title: Regression by XGBoost 基于XGBoost的回归预测实践
date: 2019-05-18 07:52:19
tags: ["Machine Learning","xgboost"]
categories: Technic
---

# 问题的提出

问题来自于Kaggle的一个比赛项目：[房价预测](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)。
给出房子的众多特征，要求建立数值回归模型，预测房子的价格。

本文完整代码在[此](https://github.com/DorianZi/kaggle/blob/master/house_prices/house_prices.py)

# 数据集

到[此处](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)下载

训练数据长这个样子：
```
Id  MSSubClass  MSZoning  LotFrontage  LotArea  Street   ...   MoSold  YrSold  SaleType  SaleCondition  SalePrice
1      60          RL         65        8450     Pave    ...     2      2008      WD        Normal       208500
2      20          RL         80        9600     Pave    ...     5      2007      WD        Normal       181500
3      60          RL         68       11250     Pave    ...     9      2008      WD        Normal       223500
4      70          RL         60        9550     Pave    ...     2      2006      WD       Abnorml       140000
5      60          RL         84       14260     Pave    ...    12      2008      WD        Normal       250000
6      50          RL         85       14115     Pave    ...    10      2009      WD        Normal       143000
7      20          RL         75       10084     Pave    ...     8      2007      WD        Normal       307000
8      60          RL         NA       10382     Pave    ...    11      2009      WD        Normal       200000
9      50          RM         51        6120     Pave    ...     4      2008      WD       Abnorml       129900
10    190          RL         50        7420     Pave    ...     1      2008      WD        Normal       118000
...
```
一共有81项特征，有数值特征，有类型特征.
特征解释(来自[官方](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data))：
```
1MSSubClass: Identifies the type of dwelling involved in the sale.

    20  1-STORY 1946 & NEWER ALL STYLES
    30  1-STORY 1945 & OLDER
    40  1-STORY W/FINISHED ATTIC ALL AGES
    45  1-1/2 STORY - UNFINISHED ALL AGES
    50  1-1/2 STORY FINISHED ALL AGES
    60  2-STORY 1946 & NEWER
    70  2-STORY 1945 & OLDER
    75  2-1/2 STORY ALL AGES
    80  SPLIT OR MULTI-LEVEL
    85  SPLIT FOYER
    90  DUPLEX - ALL STYLES AND AGES
   120  1-STORY PUD (Planned Unit Development) - 1946 & NEWER
   150  1-1/2 STORY PUD - ALL AGES
   160  2-STORY PUD - 1946 & NEWER
   180  PUD - MULTILEVEL - INCL SPLIT LEV/FOYER
   190  2 FAMILY CONVERSION - ALL STYLES AND AGES
MSZoning: Identifies the general zoning classification of the sale.

   A    Agriculture
   C    Commercial
   FV   Floating Village Residential
   I    Industrial
   RH   Residential High Density
   RL   Residential Low Density
   RP   Residential Low Density Park 
   RM   Residential Medium Density
LotFrontage: Linear feet of street connected to property

LotArea: Lot size in square feet

Street: Type of road access to property

   Grvl Gravel  
   Pave Paved
Alley: Type of alley access to property

   Grvl Gravel
   Pave Paved
   NA   No alley access
LotShape: General shape of property

   Reg  Regular 
   IR1  Slightly irregular
   IR2  Moderately Irregular
   IR3  Irregular
LandContour: Flatness of the property

   Lvl  Near Flat/Level 
   Bnk  Banked - Quick and significant rise from street grade to building
   HLS  Hillside - Significant slope from side to side
   Low  Depression
Utilities: Type of utilities available

   AllPub   All public Utilities (E,G,W,& S)    
   NoSewr   Electricity, Gas, and Water (Septic Tank)
   NoSeWa   Electricity and Gas Only
   ELO  Electricity only    
LotConfig: Lot configuration

   Inside   Inside lot
   Corner   Corner lot
   CulDSac  Cul-de-sac
   FR2  Frontage on 2 sides of property
   FR3  Frontage on 3 sides of property
LandSlope: Slope of property

   Gtl  Gentle slope
   Mod  Moderate Slope  
   Sev  Severe Slope
Neighborhood: Physical locations within Ames city limits

   Blmngtn  Bloomington Heights
   Blueste  Bluestem
   BrDale   Briardale
   BrkSide  Brookside
   ClearCr  Clear Creek
   CollgCr  College Creek
   Crawfor  Crawford
   Edwards  Edwards
   Gilbert  Gilbert
   IDOTRR   Iowa DOT and Rail Road
   MeadowV  Meadow Village
   Mitchel  Mitchell
   Names    North Ames
   NoRidge  Northridge
   NPkVill  Northpark Villa
   NridgHt  Northridge Heights
   NWAmes   Northwest Ames
   OldTown  Old Town
   SWISU    South & West of Iowa State University
   Sawyer   Sawyer
   SawyerW  Sawyer West
   Somerst  Somerset
   StoneBr  Stone Brook
   Timber   Timberland
   Veenker  Veenker
Condition1: Proximity to various conditions

   Artery   Adjacent to arterial street
   Feedr    Adjacent to feeder street   
   Norm Normal  
   RRNn Within 200' of North-South Railroad
   RRAn Adjacent to North-South Railroad
   PosN Near positive off-site feature--park, greenbelt, etc.
   PosA Adjacent to postive off-site feature
   RRNe Within 200' of East-West Railroad
   RRAe Adjacent to East-West Railroad
Condition2: Proximity to various conditions (if more than one is present)

   Artery   Adjacent to arterial street
   Feedr    Adjacent to feeder street   
   Norm Normal  
   RRNn Within 200' of North-South Railroad
   RRAn Adjacent to North-South Railroad
   PosN Near positive off-site feature--park, greenbelt, etc.
   PosA Adjacent to postive off-site feature
   RRNe Within 200' of East-West Railroad
   RRAe Adjacent to East-West Railroad
BldgType: Type of dwelling

   1Fam Single-family Detached  
   2FmCon   Two-family Conversion; originally built as one-family dwelling
   Duplx    Duplex
   TwnhsE   Townhouse End Unit
   TwnhsI   Townhouse Inside Unit
HouseStyle: Style of dwelling

   1Story   One story
   1.5Fin   One and one-half story: 2nd level finished
   1.5Unf   One and one-half story: 2nd level unfinished
   2Story   Two story
   2.5Fin   Two and one-half story: 2nd level finished
   2.5Unf   Two and one-half story: 2nd level unfinished
   SFoyer   Split Foyer
   SLvl Split Level
OverallQual: Rates the overall material and finish of the house

   10   Very Excellent
   9    Excellent
   8    Very Good
   7    Good
   6    Above Average
   5    Average
   4    Below Average
   3    Fair
   2    Poor
   1    Very Poor
OverallCond: Rates the overall condition of the house

   10   Very Excellent
   9    Excellent
   8    Very Good
   7    Good
   6    Above Average   
   5    Average
   4    Below Average   
   3    Fair
   2    Poor
   1    Very Poor
YearBuilt: Original construction date

YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)

RoofStyle: Type of roof

   Flat Flat
   Gable    Gable
   Gambrel  Gabrel (Barn)
   Hip  Hip
   Mansard  Mansard
   Shed Shed
RoofMatl: Roof material

   ClyTile  Clay or Tile
   CompShg  Standard (Composite) Shingle
   Membran  Membrane
   Metal    Metal
   Roll Roll
   Tar&Grv  Gravel & Tar
   WdShake  Wood Shakes
   WdShngl  Wood Shingles
Exterior1st: Exterior covering on house

   AsbShng  Asbestos Shingles
   AsphShn  Asphalt Shingles
   BrkComm  Brick Common
   BrkFace  Brick Face
   CBlock   Cinder Block
   CemntBd  Cement Board
   HdBoard  Hard Board
   ImStucc  Imitation Stucco
   MetalSd  Metal Siding
   Other    Other
   Plywood  Plywood
   PreCast  PreCast 
   Stone    Stone
   Stucco   Stucco
   VinylSd  Vinyl Siding
   Wd Sdng  Wood Siding
   WdShing  Wood Shingles
Exterior2nd: Exterior covering on house (if more than one material)

   AsbShng  Asbestos Shingles
   AsphShn  Asphalt Shingles
   BrkComm  Brick Common
   BrkFace  Brick Face
   CBlock   Cinder Block
   CemntBd  Cement Board
   HdBoard  Hard Board
   ImStucc  Imitation Stucco
   MetalSd  Metal Siding
   Other    Other
   Plywood  Plywood
   PreCast  PreCast
   Stone    Stone
   Stucco   Stucco
   VinylSd  Vinyl Siding
   Wd Sdng  Wood Siding
   WdShing  Wood Shingles
MasVnrType: Masonry veneer type

   BrkCmn   Brick Common
   BrkFace  Brick Face
   CBlock   Cinder Block
   None None
   Stone    Stone
MasVnrArea: Masonry veneer area in square feet

ExterQual: Evaluates the quality of the material on the exterior

   Ex   Excellent
   Gd   Good
   TA   Average/Typical
   Fa   Fair
   Po   Poor
ExterCond: Evaluates the present condition of the material on the exterior

   Ex   Excellent
   Gd   Good
   TA   Average/Typical
   Fa   Fair
   Po   Poor
Foundation: Type of foundation

   BrkTil   Brick & Tile
   CBlock   Cinder Block
   PConc    Poured Contrete 
   Slab Slab
   Stone    Stone
   Wood Wood
BsmtQual: Evaluates the height of the basement

   Ex   Excellent (100+ inches) 
   Gd   Good (90-99 inches)
   TA   Typical (80-89 inches)
   Fa   Fair (70-79 inches)
   Po   Poor (&lt;70 inches
   NA   No Basement
BsmtCond: Evaluates the general condition of the basement

   Ex   Excellent
   Gd   Good
   TA   Typical - slight dampness allowed
   Fa   Fair - dampness or some cracking or settling
   Po   Poor - Severe cracking, settling, or wetness
   NA   No Basement
BsmtExposure: Refers to walkout or garden level walls

   Gd   Good Exposure
   Av   Average Exposure (split levels or foyers typically score average or above)  
   Mn   Mimimum Exposure
   No   No Exposure
   NA   No Basement
BsmtFinType1: Rating of basement finished area

   GLQ  Good Living Quarters
   ALQ  Average Living Quarters
   BLQ  Below Average Living Quarters   
   Rec  Average Rec Room
   LwQ  Low Quality
   Unf  Unfinshed
   NA   No Basement
BsmtFinSF1: Type 1 finished square feet

BsmtFinType2: Rating of basement finished area (if multiple types)

   GLQ  Good Living Quarters
   ALQ  Average Living Quarters
   BLQ  Below Average Living Quarters   
   Rec  Average Rec Room
   LwQ  Low Quality
   Unf  Unfinshed
   NA   No Basement
BsmtFinSF2: Type 2 finished square feet

BsmtUnfSF: Unfinished square feet of basement area

TotalBsmtSF: Total square feet of basement area

Heating: Type of heating

   Floor    Floor Furnace
   GasA Gas forced warm air furnace
   GasW Gas hot water or steam heat
   Grav Gravity furnace 
   OthW Hot water or steam heat other than gas
   Wall Wall furnace
HeatingQC: Heating quality and condition

   Ex   Excellent
   Gd   Good
   TA   Average/Typical
   Fa   Fair
   Po   Poor
CentralAir: Central air conditioning

   N    No
   Y    Yes
Electrical: Electrical system

   SBrkr    Standard Circuit Breakers & Romex
   FuseA    Fuse Box over 60 AMP and all Romex wiring (Average) 
   FuseF    60 AMP Fuse Box and mostly Romex wiring (Fair)
   FuseP    60 AMP Fuse Box and mostly knob & tube wiring (poor)
   Mix  Mixed
1stFlrSF: First Floor square feet

2ndFlrSF: Second floor square feet

LowQualFinSF: Low quality finished square feet (all floors)

GrLivArea: Above grade (ground) living area square feet

BsmtFullBath: Basement full bathrooms

BsmtHalfBath: Basement half bathrooms

FullBath: Full bathrooms above grade

HalfBath: Half baths above grade

Bedroom: Bedrooms above grade (does NOT include basement bedrooms)

Kitchen: Kitchens above grade

KitchenQual: Kitchen quality

   Ex   Excellent
   Gd   Good
   TA   Typical/Average
   Fa   Fair
   Po   Poor
TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)

Functional: Home functionality (Assume typical unless deductions are warranted)

   Typ  Typical Functionality
   Min1 Minor Deductions 1
   Min2 Minor Deductions 2
   Mod  Moderate Deductions
   Maj1 Major Deductions 1
   Maj2 Major Deductions 2
   Sev  Severely Damaged
   Sal  Salvage only
Fireplaces: Number of fireplaces

FireplaceQu: Fireplace quality

   Ex   Excellent - Exceptional Masonry Fireplace
   Gd   Good - Masonry Fireplace in main level
   TA   Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
   Fa   Fair - Prefabricated Fireplace in basement
   Po   Poor - Ben Franklin Stove
   NA   No Fireplace
GarageType: Garage location

   2Types   More than one type of garage
   Attchd   Attached to home
   Basment  Basement Garage
   BuiltIn  Built-In (Garage part of house - typically has room above garage)
   CarPort  Car Port
   Detchd   Detached from home
   NA   No Garage
GarageYrBlt: Year garage was built

GarageFinish: Interior finish of the garage

   Fin  Finished
   RFn  Rough Finished  
   Unf  Unfinished
   NA   No Garage
GarageCars: Size of garage in car capacity

GarageArea: Size of garage in square feet

GarageQual: Garage quality

   Ex   Excellent
   Gd   Good
   TA   Typical/Average
   Fa   Fair
   Po   Poor
   NA   No Garage
GarageCond: Garage condition

   Ex   Excellent
   Gd   Good
   TA   Typical/Average
   Fa   Fair
   Po   Poor
   NA   No Garage
PavedDrive: Paved driveway

   Y    Paved 
   P    Partial Pavement
   N    Dirt/Gravel
WoodDeckSF: Wood deck area in square feet

OpenPorchSF: Open porch area in square feet

EnclosedPorch: Enclosed porch area in square feet

3SsnPorch: Three season porch area in square feet

ScreenPorch: Screen porch area in square feet

PoolArea: Pool area in square feet

PoolQC: Pool quality

   Ex   Excellent
   Gd   Good
   TA   Average/Typical
   Fa   Fair
   NA   No Pool
Fence: Fence quality

   GdPrv    Good Privacy
   MnPrv    Minimum Privacy
   GdWo Good Wood
   MnWw Minimum Wood/Wire
   NA   No Fence
MiscFeature: Miscellaneous feature not covered in other categories

   Elev Elevator
   Gar2 2nd Garage (if not described in garage section)
   Othr Other
   Shed Shed (over 100 SF)
   TenC Tennis Court
   NA   None
MiscVal: $Value of miscellaneous feature

MoSold: Month Sold (MM)

YrSold: Year Sold (YYYY)

SaleType: Type of sale

   WD   Warranty Deed - Conventional
   CWD  Warranty Deed - Cash
   VWD  Warranty Deed - VA Loan
   New  Home just constructed and sold
   COD  Court Officer Deed/Estate
   Con  Contract 15% Down payment regular terms
   ConLw    Contract Low Down payment and low interest
   ConLI    Contract Low Interest
   ConLD    Contract Low Down
   Oth  Other
SaleCondition: Condition of sale

   Normal   Normal Sale
   Abnorml  Abnormal Sale -  trade, foreclosure, short sale
   AdjLand  Adjoining Land Purchase
   Alloca   Allocation - two linked properties with separate deeds, typically condo with a garage unit  
   Family   Sale between family members
   Partial  Home was not completed when last assessed (associated with New Homes)
```

导入数据
```
import pandas as pd

trainDF = pd.read_csv("train.csv")
testDF = pd.read_csv("test.csv")
```

# 特征工程

## 查看价格整体分布

先看一下价格的整体分布
```
import seaborn as sns
import matplotlib.pyplot as plt

# 画密度图，也可以用更底层的plt.hist
(trainDF['SalePrice'])来做
sns.distplot(trainDF['SalePrice'])

# 因为seaborn是matplotlib的高级库，所以可以用plt.show()来调动绘图
plt.show()
```
使用sns.distplot
![](/uploads/RegressionbyXGBoost_1.png)

使用plt.hist
![](/uploads/RegressionbyXGBoost_2.png)

## 查看相关系数

```
corrmat = trainDF.corr()
print corrmat

###打印如下###
#                    Id  MSSubClass  LotFrontage   LotArea  OverallQual     ...
# Id             1.000000    0.011156    -0.010601 -0.033226    -0.028365   ...
# MSSubClass     0.011156    1.000000    -0.386347 -0.139781     0.032628   ...
# LotFrontage   -0.010601   -0.386347     1.000000  0.426095     0.251646   ...
# LotArea       -0.033226   -0.139781     0.426095  1.000000     0.105806   ...
# OverallQual   -0.028365    0.032628     0.251646  0.105806     1.000000   ...
# OverallCond    0.012609   -0.059316    -0.059213 -0.005636    -0.091932   ...
...

# 得到saleprice相对系数前十大的数据 
corrmat = strainDF.corr()
cols = corrmat.nlargest(10,'SalePrice').index
largest_price = trainDF[cols].corr()

# 绘制这前十的相关系数的热点图,其中annot=True表示将数值写入格子
sns.heatmap(largest_price,annot=True, xticklabels=largest_price.columns,yticklabels=largest_price.index)
plt.show()

print corrmat.nlargets(10,'SalePrice')
###打印如下###
#               ScreenPorch  PoolArea   MiscVal    MoSold    YrSold  SalePrice  
# SalePrice        0.111447  0.092404 -0.021190  0.046432 -0.028923   1.000000  
# OverallQual      0.064886  0.065166 -0.031406  0.070815 -0.027347   0.790982  
# GrLivArea        0.101510  0.170205 -0.002416  0.050240 -0.036526   0.708624  
# GarageCars       0.050494  0.020934 -0.043080  0.040522 -0.039117   0.640409  
# GarageArea       0.051412  0.061047 -0.027400  0.027974 -0.027378   0.623431  
# TotalBsmtSF      0.084489  0.126053 -0.018479  0.013196 -0.014969   0.613581  
# 1stFlrSF         0.088758  0.131525 -0.021096  0.031372 -0.013604   0.605852  
# FullBath        -0.008106  0.049604 -0.014290  0.055872 -0.019669   0.560664  
# TotRmsAbvGrd     0.059383  0.083757  0.024763  0.036907 -0.034516   0.533723  
# YearBuilt       -0.050364  0.004950 -0.034383  0.012398 -0.013618   0.522897  
```

![](/uploads/RegressionbyXGBoost_3.png)

## 去除异常值

先查看所有特征对于价格的散点图，查看哪些是异常值
```
for var in trainDF.select_dtypes(include=[np.number]).columns:
    plt.scatter(trainDF[var],trainDF['SalePrice'])
    plt.xlabel(var)
    plt.ylabel('SalePrice')
    plt.show()
```
随机展示几张图片：

![](/uploads/RegressionbyXGBoost_4.png)

![](/uploads/RegressionbyXGBoost_5.png)

![](/uploads/RegressionbyXGBoost_6.png)

总结起来，以下的特征的取值范围中price的值异常，需要去除该数据：

```
trainDF = trainDF.drop(trainDF[(trainDF['GrLivArea']>4000) & (trainDF['SalePrice']<300000)].index) # 默认axis=0也就是删除行
trainDF = trainDF.drop(trainDF[(trainDF['LotFrontage']>300) & (trainDF['SalePrice']<300000)].index)
trainDF = trainDF.drop(trainDF[(trainDF['BsmtFinSF1']>5000) & (trainDF['SalePrice']<200000)].index)
trainDF = trainDF.drop(trainDF[(trainDF['TotalBsmtSF']>6000) & (trainDF['SalePrice']<200000)].index)
trainDF = trainDF.drop(trainDF[(trainDF['1stFlrSF']>4000) & (trainDF['SalePrice']<200000)].index)
```

## 合并训练和测试数据

```
# 也可以直接pd.concat([trainDF,testDF],axis=0).drop(["SalePrice"])
# 因为column数不相等的两个DF也是可以拼接的，只是会产生大量NaN数据
allDF = pd.concat([trainDF.drop(["SalePrice"],axis=1),testDF],axis=0)
```

## 处理缺失值

先看一下哪些值是缺失的，再删除缺失值超过80%的特征

```
ratio = (allDF.isnull().sum()/len(allDF)).sort_values(ascending=False)
print ratio

###打印如下###
# PoolQC           0.996914
# MiscFeature      0.963992
# Alley            0.932099
# Fence            0.804184
# FireplaceQu      0.486968
# LotFrontage      0.166667
# GarageQual       0.054527
# GarageFinish     0.054527
# ...

allDF.drop(ratio[ratio>0.8].index,axis=1)
```

对未删除的特征，进行缺失值处理

```
allDF["LotFrontage"] = allDF.groupby("Neighborhood")["LotFrontage"].transform(
lambda x: x.fillna(x.median()))

for col in ('FireplaceQu',\
    'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', \
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',\
    'MasVnrType',\
    'MSSubClass'):
    allDF[col] = allDF[col].fillna('None')

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars',\
    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath',\
    'MasVnrArea'):
    allDF[col] = allDF[col].fillna(0)

# 因为数值都一样，没有存在意义
allDF = allDF.drop(['Utilities'], axis=1)

for col in ('MSZoning','Functional','Electrical','KitchenQual',\
    'Exterior1st','Exterior2nd','SaleType'):
    allDF[col] = allDF[col].fillna(allDF[col].mode()[0])

```

## 离散值转换

有些数值看起来是连续值，但是上它们的取值只是在有限数值上变动，因此可以变成离散值。方法就是将数值类型转换为字符串类型。
但是注意，最终我们需要将字符串通过哑变量转换为数值类型，以便给XGBoost计算

```
# 也可以用 allDF['MSSubClass'] = allDF['MSSubClass'].transform(lambda x:str(x))
allDF['MSSubClass'] = allDF['MSSubClass'].apply(str)   # 应用到每一个元素
allDF['OverallCond'] = allDF['OverallCond'].astype(str)
allDF['YrSold'] = allDF['YrSold'].astype(str)
allDF['MoSold'] = allDF['MoSold'].astype(str)
```

## 创造新特征

```
allDF['TotalSF'] = allDF['TotalBsmtSF'] + allDF['1stFlrSF'] + allDF['2ndFlrSF']
```

## 转换为哑变量

```
dm_allDF = pd.get_dummies(allDF)
```

## 训练和预测

```
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# 重新分割训练和测试数据
dm_trainDF = dm_allDF[:len(trainDF)]
dm_testDF = dm_allDF[len(trainDF):]

# 去掉id号
train_data = dm_trainDF.drop(['Id'],axis=1).values
train_label = trainDF_label.values

X_test_ids = dm_testDF['Id'].values
X_test = dm_testDF.drop(['Id'],axis=1).values

# 分割训练集和验证集
X_train, X_valid, Y_train, Y_valid = train_test_split(train_data, train_label,test_size=0.2)

xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, min_child_weight=5, max_depth=4)
xgb.fit(X_train,Y_train)
print "Validation:",xgb.score(X_valid,Y_valid)

predict = xgb.predict(X_test)

```

验证结果示例

```
Validation: 0.92924263948
```

以上。
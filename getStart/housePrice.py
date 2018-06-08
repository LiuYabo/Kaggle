import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette() #设置seaborn的调色板，更改循环色
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn()
# warnings.filterwarnings('ignore')

from scipy import stats
from scipy.stats import norm, skew
# pd.set_option("display.float_format", lambda x:'{:,3f}'.format(x))

from subprocess import check_output

train = pd.read_csv('data/housePrice/train.csv')
test = pd.read_csv("data/housePrice/test.csv")

train.info()

train_Id = train.Id
test_Id = test.Id
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)

print('The train data size without id is,{}'.format(train.shape))
print('The test data size without id is,{}'.format(test.shape))

# fig, ax = plt.subplots()
# ax.scatter(x=train.GrLivArea, y=train.SalePrice)
# plt.ylabel('SalePrice')
# plt.xlabel('GrLivArea')

#放弃不合理点
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

# fig, ax = plt.subplots()
# ax.scatter(x=train.GrLivArea, y=train.SalePrice)
# plt.ylabel('SalePrice')
# plt.xlabel('GrLivArea')

# sns.distplot(train['SalePrice'], fit=norm)

# fig = plt.figure()
# res = stats.probplot(train['SalePrice'], plot=plt)


#saleprice log转换
train.SalePrice = np.log1p(train.SalePrice)
# sns.distplot(train['SalePrice'])
train['SalePrice'].plot(kind='kde')


# (mu, sigma) = norm.fit(train['SalePrice'])
# print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
#
# #Now plot the distribution
# plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
#             loc='best')
# plt.ylabel('Frequency')
# plt.title('SalePrice distribution')

# #Get also the QQ-plot
# fig = plt.figure()
# res = stats.probplot(train['SalePrice'], plot=plt)

#处理数据缺失值
#train与test同时处理
ntrain = train.shape[0]
ntest = test.shape[0]

y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)  #reset_index，重新设置索引
all_data.drop('SalePrice', axis=1, inplace=True)  #inplace表示直接在原对象操作，不产生新对象
print('all data size is :{}'.format(all_data.shape))

all_data_na = (all_data.isnull().sum()/len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]   #取空值前30多的特征
missing_data = pd.DataFrame({'Missing Ratio':all_data_na})
print(missing_data.head(30))

#显示各个特征的缺失值多少
f, ax = plt.subplots(figsize=(15,12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features')
plt.ylabel('percent of missing values')
plt.title('percent missing data by feature')

corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)

#填充缺失值
all_data.PoolQC = all_data['PoolQC'].fillna('None') #None为python数据类型，NaN为numpy pandas的空值，实际为float类型











plt.show()








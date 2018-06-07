import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df_train = pd.read_csv('data/housePrice/train.csv')
# print(df_train.columns)

# print(df_train['SalePrice'].describe())
df_train.info()

# fig = plt.figure()
# df_train.SalePrice.plot(kind='kde')


print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())



# data = pd.concat([df_train.SalePrice, df_train.GrLivArea], axis=1)
# data.plot.scatter()

# var = 'GrLivArea'
# data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
# data.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0,800000));
#
# data = pd.concat([df_train.SalePrice, df_train.TotalBsmtSF], axis=1)
# data.plot.scatter('TotalBsmtSF', 'SalePrice', ylim=(0,800000))
#
# var = 'OverallQual'
# data = pd.concat([df_train.OverallQual, df_train.SalePrice], axis=1)
# f, ax=plt.subplots(figsize=(8,6))
# fig = sns.boxplot(var, 'SalePrice', data=data)
# fig.axis(ymin=0, ymax=80000)
#
# var = 'YearBuilt'
# data = pd.concat([df_train.SalePrice, df_train[var]], axis=1)
# f, ax = plt.subplots(figsize=(16,8))
# fig = sns.boxplot(var, 'SalePrice', data=data)
# fig.axis(ymin=0, ymax=800000)
# plt.xticks(rotation=90)

# #相关矩阵
# corrmat = df_train.corr()
# f, ax = plt.subplots(figsize=(12,9))
# sns.heatmap(corrmat, vmax=0.8, square=True)
#
# #售价相关矩阵
# k = 10
# cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
# cm = np.corrcoef(df_train[cols].values.T)
# sns.set(font_scale=1.25)
# hm = sns.heatmap(cm, cbar=True, square=True, fmt='.2f', annot_kws={'size':10},yticklabels=cols.values, xticklabels=cols.values)
#
# sns.set()
# cols=['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
# sns.pairplot(df_train[cols], size=2.5)

total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(20))
















plt.show()




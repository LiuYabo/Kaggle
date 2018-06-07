import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
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

var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter('GrLivArea', y='SalePrice', ylim=(0,800000));

plt.show()




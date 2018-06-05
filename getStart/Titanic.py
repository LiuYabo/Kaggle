# _*_ coding=utf-8 _*_
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt

data_train = pd.read_csv('data/Titanic/train.csv')
# data_train.info()
# data_train.describe()

fig = plt.figure()
fig.set(alpha=0.2)

plt.subplot2grid((2,3), (0,0))
data_train.Survived.value_counts().plot(kind='bar')
plt.title('获救情况')
plt.ylabel('人数')
print(data_train.Survived.value_counts())

plt.subplot2grid((2,3), (0,1))
data_train.Pclass.value_counts().plot(kind='bar')
plt.title('等级分布情况')
plt.ylabel('人数')

plt.subplot2grid((2,3), (0,2))
plt.scatter(data_train.Survived, data_train.Age)
plt.title('各年龄获救情况')
plt.ylabel('年龄')

plt.subplot2grid((2,3), (1,0), colspan=2)
data_train.Age[data_train.Pclass==1].plot(kind='kde')
data_train.Age[data_train.Pclass==2].plot(kind='kde')
data_train.Age[data_train.Pclass==3].plot(kind='kde')
plt.title('各等级乘客年龄分布')
plt.legend(('头等舱','二等舱','三等舱'), loc='best')

plt.subplot2grid((2,3), (1,2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title('各口岸上船人数')
plt.ylabel('人数')

fig = plt.figure()
survived_0 = data_train.Pclass[data_train.Survived==0].value_counts()
survived_1 = data_train[data_train.Survived==1].Pclass.value_counts()
df = pd.DataFrame({'获救':survived_1, '未获救':survived_0})
df.plot(kind='bar', stacked=True)
plt.title('各等级获救情况')
plt.ylabel('人数')

fig = plt.figure()
survived_0 = data_train[data_train.Sex=='male'].Survived.value_counts()
survived_1 = data_train[data_train.Sex=='Female'].Survived.value_counts()
df = pd.DataFrame({'获救':survived_1, '未获救':survived_0})
df.plot(kind='bar', stacked=True)

# plt.show()

from sklearn.ensemble import RandomForestRegressor
def set_missing_age(df):
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    know_age = age_df[age_df.Age.notnull()].as_matrix()
    unknow_age = age_df[age_df.Age.isnull()].as_matrix()

    y = know_age[:,0]
    x = know_age[:,1:]

    rfr = RandomForestRegressor(n_estimators=200)
    rfr.fit(x, y)
    predictions = rfr.predict(unknow_age[:,1:])

    df.loc[df.Age.isnull(), 'Age'] = predictions
    return df, rfr

def set_Cabin_type(df):
    df.loc[df.Cabin.isnull(), 'Cabin'] = 'no'
    df.loc[df.Cabin.notnull(), 'Cabin'] = 'yes'
    return df

data_train, rfr = set_missing_age(data_train)
data_train = set_Cabin_type(data_train)
data_train.loc[data_train['Embarked'].isnull(),'Embarked'] = 'S'

#因子化，将无意义的分类属性，区分为多个属性
dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix='Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')
df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)


import sklearn.preprocessing as preprocessing

# scaler = preprocessing.StandardScaler()
# age_scale_parm = scaler.fit(df.Age)
# fare_scale_parm = scaler.fit(df.Fare)
# df.Age = scaler.fit_transform(df.Age, age_scale_parm)
# df.Fare = scaler.fit_transform(df.Fare, fare_scale_parm)

scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df['Age'])
df['Age_scaled'] = scaler.fit_transform(df['Age'], age_scale_param)
fare_scale_param = scaler.fit(df['Fare'])
df['Fare_scaled'] = scaler.fit_transform(df['Fare'], fare_scale_param)

from sklearn.linear_model import LogisticRegression
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# train_df = df.filter(regex='Survived|Age|SibSp|Parch|Fare|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.as_matrix()

y = train_np[:,0]
x = train_np[:,1:]

clf = LogisticRegression(C=1.0, penalty='l1',tol=1e-6)
clf.fit(x,y)

#验证集数据预处理
data_test = pd.read_csv('data/Titanic/test.csv')
data_test.loc[data_test.Fare.isnull(), 'Fare'] = 0
tem_df = data_test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tem_df[tem_df.Age.isnull()].as_matrix()

x_null_age = null_age[:,1:]
predictions_age = rfr.predict(x_null_age)
data_test.loc[data_test.Age.isnull(), 'Age'] = predictions_age

data_test = set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')

df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'], age_scale_param)
df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'], fare_scale_param)


test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
test_np = test.as_matrix()
test.info()
predictions = clf.predict(test_np)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
result.to_csv("output/Titanic/logistic_regression_predictions.csv", index=False)


#模型系数与特征关联
pd.DataFrame({'columns':list(train_df.columns)[1:], 'coef':list(clf.coef_.T)})



#下一步完成交叉验证和ensemble集合提升













































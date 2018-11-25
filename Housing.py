import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler

# import seaborn as sns
names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
boston = pd.read_csv('./housing.txt', sep="\s+", header=None, names=names)

# Data without Median House Value that we want regressor to predict
boston_train = boston.loc[:, boston.columns != 'MEDV']
# Targets that we want the Regression Machine to predict
boston_target = boston.loc[:, boston.columns == 'MEDV']

# Check if data is empty anywhere
print(boston_train.isnull().sum())

#Check if labels are null anywhere
print(boston_target.isnull().sum())
# All the data is filled in (no NAN in dataset)

# Plot distribution of the target variable MEDV
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7, 8.27)})
sns.distplot(boston_target, bins=30)
plt.show()
#Plot shows that there are very few outliers in the Target Data

# use a corr matrix to see the relationships between the variables
# Use corr() from pandas and heatmap from seaborn
correlation_matrix = boston.corr().round(2)
#annot = TRUE, prints the correlation valus in squares
sns.heatmap(data=correlation_matrix, annot=True)
plt.show()
# Since I want to predict MEDV with great accuracy, I should
# Only use features that correlate with MEDV greatly

#From the informative Heat Map, we know RM and LSTAT are highly correlated to MEDV
# So we use a scatter plot to see how these features vary with MEDV
plt.figure(figsize=(20,5))

features = ['LSTAT', 'RM', 'PTRATIO', 'INDUS']
target = boston_target

for i, col in enumerate(features):
    plt.subplot(1, len(features), i+1)
    x = boston_train[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('MEDV')
plt.show()
# PRICES INCREASE AS RM INCREASE LINEARLY & CAPPED AT
# PRICES DECREASE AS LSTAT INCREASE IN SOMETHING NOT EXACTLY LINEAR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

X = pd.DataFrame(np.c_[boston_train['LSTAT'],boston['RM'], boston['INDUS'], boston['PTRATIO']], columns=['LSTAT', 'RM', 'INDUS','PTRATIO'])
Y = boston_target


X_train,X_test, y_train,y_test = train_test_split(X, y, test_size=.3, random_state=42)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

lin_model = LinearRegression()
lin_model.fit(X, y)

# model evaluation for training set

y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
r2 = r2_score(y_train, y_train_predict)



print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
# print('Accuracy: ', np.where(y_train_predict != y_train, 1, 0).sum()/len(y_train))
print("\n")



# model evaluation for testing set

y_test_predict = lin_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
r2 = r2_score(y_test, y_test_predict)



print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
# # print('Accuracy: ', np.where(y_test_predict != y_test, 1, 0).sum()/len(y_test))

'-------------------------------------------------------------------------------------------------------------'
# Quadratic Curve Performance
lr2 = LinearRegression()
quadratic = PolynomialFeatures(degree=2)
X_quad = quadratic.fit_transform(X_train)

lr2.fit(X_quad, y_train)
y_quad_fit = lr2.predict(quadratic.fit_transform(X_test))

y_quad_pred = lr2.predict(X_quad)
print('Training MSE quadratic: %.3f' % (mean_squared_error(y_train, y_quad_pred)))
print('Training R^2 quadratic: %.3f' % (r2_score(y_train, y_quad_pred)))

lr3 = LinearRegression()
quadratic = PolynomialFeatures(degree=2)
X_quad = quadratic.fit_transform(X_test)

lr3.fit(X_quad, y_test)
y_quad_fit = lr3.predict(quadratic.fit_transform(X_test))

y_quad_pred = lr3.predict(X_quad)
print('Testing MSE quadratic: %.3f' % (mean_squared_error(y_test, y_quad_pred)))
print('Testing R^2 quadratic: %.3f' % (r2_score(y_test, y_quad_pred)))
print('\n\n')
'---------------------------------------------------------------------------------------------------------------'
# Regularizing Methods for Regression
# Basic Lasso
from sklearn.linear_model import Lasso, Ridge,ElasticNet
Ls = Lasso(alpha=1.0)
Ls.fit(X_train, y_train)
Las_pred = Ls.predict(X_train)
print('Training MSE Lasso: %.3f' % (mean_squared_error(y_train, Las_pred)))
print('Training R^2 Lasso: %.3f' % (r2_score(y_train, Las_pred)))

Ls = Lasso(alpha=5.9)
Ls.fit(X_train, y_train)
Las_pred = Ls.predict(X_train)
print('Training MSE Lasso: %.3f' % (mean_squared_error(y_train, Las_pred)))
print('Training R^2 Lasso: %.3f' % (r2_score(y_train, Las_pred)))

Ls = Lasso(alpha=0.001)
Ls.fit(X_train, y_train)
Las_pred = Ls.predict(X_train)
print('Training MSE Lasso: %.3f' % (mean_squared_error(y_train, Las_pred)))
print('Training R^2 Lasso: %.3f' % (r2_score(y_train, Las_pred)))
print('\n\n')
# Basic Ridge
ridge = Ridge(alpha=0.001)
ridge.fit(X_train, y_train)
print(ridge.score(X_train,y_train))
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
print(ridge.score(X_train,y_train))
ridge = Ridge(alpha=6.0)
ridge.fit(X_train, y_train)
print(ridge.score(X_train,y_train))
print('\n\n')

#Elastic Net Regression
elanet =ElasticNet(alpha=1.0, l1_ratio=0.5)
elanet.fit(X_train, y_train)
el_pred = elanet.predict(X_train)
print('Training MSE Elastic: %.3f' % (mean_squared_error(y_train, el_pred)))
print('Training R^2 Elastic: %.3f' % (r2_score(y_train, el_pred)))

elanet =ElasticNet(alpha=0.0001, l1_ratio=0.7)
elanet.fit(X_train, y_train)
el_pred = elanet.predict(X_train)
print('Training MSE Elastic: %.3f' % (mean_squared_error(y_train, el_pred)))
print('Training R^2 Elastic: %.3f' % (r2_score(y_train, el_pred)))

elanet =ElasticNet(alpha=0.0001, l1_ratio=1.0)
elanet.fit(X_train, y_train)
el_pred = elanet.predict(X_train)
print('Training MSE Elastic: %.3f' % (mean_squared_error(y_train, el_pred)))
print('Training R^2 Elastic: %.3f' % (r2_score(y_train, el_pred)))

elanet =ElasticNet(alpha=0.0001, l1_ratio=0.1)
elanet.fit(X_train, y_train)
el_pred = elanet.predict(X_train)
print('Training MSE Elastic: %.3f' % (mean_squared_error(y_train, el_pred)))
print('Training R^2 Elastic: %.3f' % (r2_score(y_train, el_pred)))

elanet =ElasticNet(alpha=5, l1_ratio=0.5)
elanet.fit(X_train, y_train)
el_pred = elanet.predict(X_train)
print('Training MSE Elastic: %.3f' % (mean_squared_error(y_train, el_pred)))
print('Training R^2 Elastic: %.3f' % (r2_score(y_train, el_pred)))

elanet =ElasticNet(alpha=5, l1_ratio=0.9)
elanet.fit(X_train, y_train)
el_pred = elanet.predict(X_train)
print('Training MSE Elastic: %.3f' % (mean_squared_error(y_train, el_pred)))
print('Training R^2 Elastic: %.3f' % (r2_score(y_train, el_pred)))

elanet =ElasticNet(alpha=5, l1_ratio=0.1)
elanet.fit(X_train, y_train)
el_pred = elanet.predict(X_train)
print('Training MSE Elastic: %.3f' % (mean_squared_error(y_train, el_pred)))
print('Training R^2 Elastic: %.3f' % (r2_score(y_train, el_pred)))
print('\n\n')
'-----------------------------------------------------------------------------------------------------------------------'
 # Since the Qudatratic Curve has the highest correlation and low ROOT MEAN SQUARE ERROR
 # We will test the results of a svm
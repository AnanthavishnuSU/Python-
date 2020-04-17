import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_boston

boston = load_boston()  # To print the boston dataset
print(boston.DESCR)

dataset = boston.data  # Access data attributes
for name, index in enumerate(boston.feature_names):
    print(index, name)

data = dataset[:, 12].reshape(-1, 1)  # reshaping
print(np.shape(data))  # Shape of the data

target = boston.target.reshape(-1, 1)  # Target values
print(np.shape(target))  # shape of target

plt.scatter(data, target, color='red')   # To check is matplotlib is working
plt.xlabel('Lower income population')
plt.ylabel('Cost of House')
plt.show()

# Regression
from sklearn.linear_model import LinearRegression   # Linear Regression

reg = LinearRegression()  # Creating a regression model
reg.fit(data, target)  # fit model

pred = reg.predict(data)

plt.scatter(data, target, color='red')   # To see the prediction
plt.plot(data, pred, color='green')
plt.xlabel('Lower income population')
plt.ylabel('Cost of House')
plt.show()


from sklearn.linear_model import Lasso  # Lasso Regression

reg = Lasso()  # Creating a regression model
reg.fit(data, target)  # fit model

pred = reg.predict(data)

plt.scatter(data, target, color='red')   # To see the prediction
plt.plot(data, pred, color='green')
plt.xlabel('Lower income population')
plt.ylabel('Cost of House')
plt.show()


from sklearn.linear_model import Ridge  # Ridge Regression

reg = Ridge()  # Creating a regression model
reg.fit(data, target)  # fit model

pred = reg.predict(data)

plt.scatter(data, target, color='red')   # To see the prediction
plt.plot(data, pred, color='green')
plt.xlabel('Lower income population')
plt.ylabel('Cost of House')
plt.show()

# Polynomial modeling

# Circumventing curve issue using polynomial model

from sklearn.preprocessing import PolynomialFeatures

# To allow merging of model
from sklearn.pipeline import make_pipeline

model = make_pipeline(PolynomialFeatures(7), reg)

model.fit(data, target)
pred = model.predict(data)

plt.scatter(data, target, color='red')   # To see the prediction
plt.plot(data, pred, color='green')
plt.xlabel('Lower income population')
plt.ylabel('Cost of House')
plt.show()

# r_2_Metric

from sklearn.metrics import r2_score

print(r2_score(pred, target))  # r2_matrics varies between 1 and -1 the more close to 1 the more sink with the data


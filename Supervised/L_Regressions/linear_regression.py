from sklearn import linear_model
import pandas as pd
import numpy as np


data = pd.read_csv("data.csv")
data.drop(['Timestamp'], axis=1)

x = np.array(data.drop(["Open"], 1))
y = np.array(data["Open"])


train_x = x[:int((len(x))/2)]
test_x = x[int((len(x))/2):]

train_y = y[:int((len(y))/2)]
test_y = y[int((len(y))/2):]

print(f"Len of trainX: {len(train_x)}")
print(f"Len of totalX: {len(test_x)}")
print(f"Len of testY: {len(train_y)}")
print(f"Len of testY: {len(test_y)}")

print(f"Total length: {len(x)}")

linear = linear_model.LinearRegression()
linear_coeff = linear_model.LinearRegression(fit_intercept=True)

linear.fit(train_x, train_y)
linear_coeff.fit(train_x, train_y)

print(f"Score without coefficient: {linear.score(test_x, test_y)}")
print("-------------------------------------------------")
print(f"Score with coefficient: {linear_coeff.score(test_x, test_y)}")
print(f"Coefficient: {linear_coeff.coef_}")

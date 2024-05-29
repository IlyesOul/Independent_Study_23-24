from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


# data = load_breast_cancer()
#
# x = data.data
# y = data.target
#
# x_train, x_test, y_train, y_test = train_test_split(x, y)
#
# train_scores = []
# test_scores = []
#
# for i in range(4, 15, 2):
# 	d_tree = DecisionTreeClassifier(criterion="gini", max_depth=i)
# 	d_tree.fit(x_train, y_train)
# 	train_scores.append(d_tree.score(x_train, y_train))
# 	test_scores.append(d_tree.score(x_test, y_test))
#
# depths = [i for i in range(4, 15, 2)]
#
# plt.xlabel("Max Depths")
# plt.ylabel("Scores")
#
# plt.plot(depths, train_scores, c="blue", label="Training Scores")
# plt.plot(depths, test_scores, c="red", label="Testing Scores")
#
# plt.legend()
# plt.show()


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import statistics
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


# CROSS-VALIDATION

# Testing K-fold against train_test_split against Stratified K-Fold

# log = LogisticRegression(C=1)
# # Train_test_split testing
# data = load_iris()
#
# x_train, x_test, y_train, y_test = train_test_split(data.data, data.target)
#
# log.fit(x_train, y_train)
# print(f"Tran_test_split score: {log.score(x_test, y_test)}")
#
# # Cross validation testing
# three_fold = KFold(n_splits=3)
#
# cross_val_scores = cross_val_score(log, data.data, data.target, cv=three_fold)
# print(f"Cross Validation average score of {statistics.mean(cross_val_scores)} and standard deviation of {statistics.stdev(cross_val_scores)}")
#
# # Stratified K-Fold
# log = LogisticRegression(C=1)
# stratified = StratifiedKFold(n_splits=3)
# cross_val_scores = cross_val_score(log, data.data, data.target, cv=stratified)
#
# print(f"Stratified Cross Validation average score of {statistics.mean(cross_val_scores)} and standard deviation of {statistics.stdev(cross_val_scores)}")

# GRID SEARCH

# Implementing a simple grid search (O(N^2))
scalar = MinMaxScaler()
#
# x_train, x_test, y_train, y_test = train_test_split(load_iris().data, load_iris().target)
#
# x_train_scaled = scalar.fit_transform(x_train)
# x_test_scaled = scalar.fit_transform(x_test)
#
# best_score = 0
# optimal_params = {}
#
# for i in [.001, .01, .1, 1, 10, 100]:
# 	for gam in [.001, .01, .1, 1, 10, 100]:
# 		# Instantiate, train, and score tree
# 		tree = SVC(C=i, gamma=gam)
# 		tree.fit(x_train_scaled, y_train)
# 		score = tree.score(x_test_scaled, y_test)
#
# 		# Replace best score
# 		if score > best_score:
# 			best_score = score
#
# 			optimal_params = {"C":i, "gamma":gam}
#
# print(f"Best score is {best_score} with {optimal_params}")
#
# # Grid search with cross validation for scoring
# for i in [.001, .01, .1, 1, 10, 100]:
# 	for gam in [.001, .01, .1, 1, 10, 100]:
# 		# Instantiate, train, and score tree
# 		tree = SVC(C=i, gamma=gam)
# 		tree.fit(x_train_scaled, y_train)
# 		score = statistics.mean(cross_val_score(tree, x_test, y_test))
#
# 		# Replace best score
# 		if score > best_score:
# 			best_score = score
#
# 			optimal_params = {"C":i, "gamma":gam}
#
# print(f"Best score is {best_score} with {optimal_params}")
#
# optimal_tree = SVC(**optimal_params)
# optimal_tree.fit(x_train, y_train)
# print(f"Optimized tree score is {optimal_tree.score(x_test, y_test)}")


# Sklearn Implementation of GridSearchCV
x_train, x_test, y_train, y_test = train_test_split(load_iris().data, load_iris().target)

x_train_scaled = scalar.fit_transform(x_train)
x_test_scaled = scalar.fit_transform(x_test)

test_svc = SVC()
test_svc.fit(x_train_scaled, y_train)
gridsearch = GridSearchCV(test_svc, {'kernel':('linear', 'rbf'), 'C':[1, 10], 'gamma': [.001, .01, .1, 1, 10, 100]})
gridsearch.fit(x_train_scaled, y_train)

# Display GridSearchCV results

print(f"Regular score: {test_svc.score(x_test_scaled,y=y_test)}")
print(f"Optimized score: {gridsearch.score(x_test_scaled,y_test)}")


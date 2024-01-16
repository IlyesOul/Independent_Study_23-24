from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
import matplotlib.pyplot as mlp

data = load_iris()

# Scalar Experimentation
x_train, x_test, y_train, y_test = train_test_split(data.data, data.target)

# Regular
svm = SVC(C=100)
svm.fit(x_train, y_train)
print(f"Normal Score = {svm.score(x_test, y_test)}")

# MinMaxScalar
minmax = MinMaxScaler()
minmax.fit(x_train)
x_t1 = minmax.transform(x_train)
x_t2 = minmax.transform(x_test)
svm = SVC(C=100)

svm.fit(x_t1, y_train)
print(f"MinMaxScalar score = {svm.score(x_t2, y_test)}")

# Standard Scalar
minmax = StandardScaler()
minmax.fit(x_train)
x_t1 = minmax.transform(x_train)
x_t2 = minmax.transform(x_test)
svm = SVC(C=100)
svm.fit(x_t1, y_train)
print(f"StandardScalar score = {svm.score(x_t2, y_test)}")

# RobustScalar
minmax = RobustScaler()
minmax.fit(x_train)
x_t1 = minmax.transform(x_train)
x_t2 = minmax.transform(x_test)
svm = SVC(C=100)
svm.fit(x_t1, y_train)
print(f"Robust score = {svm.score(x_t2, y_test)}")

# TODO: EXPERIMENT WITH CLUSTERING ON ACTUAL STOCK DATA

minmax.fit(x_train)
x_scaled_train = minmax.transform(x_train)
x_scaled_test = minmax.transform(x_test)

# K-Means Clustering
scores = []

for i in range(10):
	k_means = KMeans(n_clusters=i)
	k_means.fit(x_scaled_train)
	scores.append(k_means.score(x_test, y_test))

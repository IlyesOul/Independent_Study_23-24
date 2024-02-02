from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, AffinityPropagation
import matplotlib.pyplot as plt
import numpy as np

data = load_iris()

# Scalar Experimentation
x_train, x_test, y_train, y_test = train_test_split(data.data, data.target)

# Regular SVM
# svm = SVC(C=100)
# svm.fit(x_train, y_train)
# print(f"Normal Score = {svm.score(x_test, y_test)}")

# MinMaxScalar
minmax = MinMaxScaler()
minmax.fit(x_train)
# x_t1 = minmax.transform(x_train)
# x_t2 = minmax.transform(x_test)
# svm = SVC(C=100)
#
# svm.fit(x_t1, y_train)
# print(f"MinMaxScalar score = {svm.score(x_t2, y_test)}")
#
# # Standard Scalar
# minmax = StandardScaler()
# minmax.fit(x_train)
# x_t1 = minmax.transform(x_train)
# x_t2 = minmax.transform(x_test)
# svm = SVC(C=100)
# svm.fit(x_t1, y_train)
# print(f"StandardScalar score = {svm.score(x_t2, y_test)}")
#
# # RobustScalar
# minmax = RobustScaler()
# minmax.fit(x_train)
# x_t1 = minmax.transform(x_train)
# x_t2 = minmax.transform(x_test)
# svm = SVC(C=100)
# svm.fit(x_t1, y_train)
# print(f"Robust score = {svm.score(x_t2, y_test)}")


minmax.fit(x_train)
x_scaled_train = minmax.transform(x_train)
x_scaled_test = minmax.transform(x_test)

# CLUSTERING


# dbscan = DBSCAN(eps=.01)
# clusters = dbscan.fit_predict(x_scaled_train)
# print(len(clusters))
# print(len(np.bincount(clusters)))

# dbscan = DBSCAN(eps=.1)
# clusters = dbscan.fit_predict(x_scaled_train)
# print(len(clusters))
# print(len(np.bincount(clusters)))

print(f"DBSCAN Clustering")

for ep in range(1, 15):
	dbscan = DBSCAN(min_samples=2, eps=ep)
	clusters = dbscan.fit_predict(x_scaled_train)
	print(f"eps = {ep}")
	print(f"Unique labels: {np.unique(clusters)}")
	print(f"# of Clusters: {len(clusters)}")
	print(f"# of points in each cluster: {np.bincount(clusters +1)}")
	print()


# Agglomerative Clustering
print(f"Average Agglomerative Clustering")
average = AgglomerativeClustering(linkage="average")
clusters = average.fit_predict(x_scaled_train)
print(f"Unique labels: {np.unique(clusters)}")
print(f"# of Clusters: {len(clusters)}")
print(f"# of points in each cluster: {np.bincount(clusters)}")
print()
print()

print(f"Ward Agglomerative Clustering")
average = AgglomerativeClustering(linkage="ward")
clusters = average.fit_predict(x_scaled_train)
print(f"Unique labels: {np.unique(clusters)}")
print(f"# of Clusters: {len(clusters)}")
print(f"# of points in each cluster: {np.bincount(clusters)}")
print()
print()

print(f"Complete Agglomerative Clustering")
average = AgglomerativeClustering(linkage="complete")
clusters = average.fit_predict(x_scaled_train)
print(f"Unique labels: {np.unique(clusters)}")
print(f"# of Clusters: {len(clusters)}")
print(f"# of points in each cluster: {np.bincount(clusters)}")
print()
print()

import matplotlib.pyplot as plt
import sklearn.model_selection
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Training dataset
cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(cancer.data, cancer.target, random_state=0)

training_accuracy = []
test_accuracy = []
tot_neighbors = range(1, 11)

# KNN Testing
for n in tot_neighbors:
    model = KNeighborsClassifier(n_neighbors=n)
    model.fit(x_train, y_train)
    training_accuracy.append(model.score(x_train, y_train))
    test_accuracy.append(model.score(x_test, y_test))

plt.title("KNN Neighbor testing")
plt.plot(tot_neighbors, training_accuracy, label="training accuracy")
plt.plot(tot_neighbors, test_accuracy, label="testing accuracy")
plt.xlabel("n_neighbors")
plt.ylabel("accuracy")
plt.legend()

plt.show()

# Ridge testing
alphas = [.00000000001, .0000000001, .000000001, .000000001, .00000001, .0000001, .000001, .00001, .0001, .001, .01, .1, 1, 10]
training_accuracy = []
test_accuracy = []

for alph in alphas:
    model = Ridge(alpha=alph)
    model.fit(x_train, y_train)
    training_accuracy.append(model.score(x_train, y_train))
    test_accuracy.append(model.score(x_test, y_test))

plt.title("Ridge Alpha-parameter testing")
plt.plot(alphas, training_accuracy, label="Training accuracy")
plt.plot(alphas, test_accuracy, label="Testing accuracy")
plt.xlabel("Alpha")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Logistic Classification

logreg = LogisticRegression(C=1).fit(x_train, y_train)
logreg01 = LogisticRegression(C=.01).fit(x_train, y_train)
logreg10 = LogisticRegression(C=100).fit(x_train, y_train)

plt.title("Logistic Regression Magnitude of Coefficients")
plt.plot(logreg.coef_.T, 'o', label="C=1")
plt.plot(logreg01.coef_.T, '^', label="C=.01")
plt.plot(logreg10.coef_.T, 'v', label="C=10")

plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.ylim(-5, 5)
plt.xlabel("Feature")
plt.ylabel("Magnitude of Coefficient")
plt.legend()
plt.show()

# Higher C values corresponds to less regularization
c_values = [.01, .1, 1, 10]

training_accuracy = []
test_accuracy = []

for val in c_values:
    model = LogisticRegression(C=val)
    model.fit(x_train, y_train)
    training_accuracy.append(model.score(x_train, y_train))
    test_accuracy.append(model.score(x_test, y_test))

plt.title("Logistic Regression C-Value testing")
plt.plot(c_values, training_accuracy, label="Training accuracy")
plt.plot(c_values, test_accuracy, label="Testing accuracy")
plt.xlabel("C")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Gaussian Classification Testing
gs = GaussianNB()
gs.fit(x_train, y_train)

print(f"Gaussian Score = {gs.score(x_test, y_test)}")

# Decision Tree Testing
tree = DecisionTreeClassifier(max_depth=5, min_samples_split=12)
tree.fit(x_train, y_train)

print(f"Training accuracy = {tree.score(x_train, y_train)}")
print(f"Testing accuracy = {tree.score(x_test, y_test)}")

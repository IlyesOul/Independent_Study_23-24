from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

X, y = load_breast_cancer(return_X_y=True)

x_train,  x_test, y_train, y_test = train_test_split(X, y)

c_vals = []
scores = []

for i in range(1, 6):
	log = LogisticRegression(C=i, penalty="l2")
	log.fit(x_train, y_train)
	c_vals.append(i)
	scores.append(log.score(x_test, y_test))

plt.title("C Accuracy Testing")
plt.plot(c_vals, scores, label="training accuracy")
plt.xlabel("C Values")
plt.ylabel("Score")
plt.show()

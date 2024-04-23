from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import statistics

log = LogisticRegression(C=1)

# Testing K-fold against train_test_split

# Train_test_split testing
data = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(data.data, data.target)

log.fit(x_train, y_train)
print(f"Tran_test_split score: {log.score(x_test, y_test)}")

# Cross validation testing
cross_val_scores = cross_val_score(log, data.data, data.target)
print(f"Cross Validation average score of {statistics.mean(cross_val_scores)} and standard deviation of {statistics.stdev(cross_val_scores)}")


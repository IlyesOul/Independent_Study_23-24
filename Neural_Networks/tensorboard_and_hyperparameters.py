from tensorflow import keras
import os
import widedeepnet
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal
import numpy as np
from scikeras.wrappers import KerasRegressor

# Data importation & preprocessing
scalar = MinMaxScaler()
data = fetch_california_housing()


# Data splitting & preprocessing
x_ftrain, x_test, y_ftrain, y_test = train_test_split(data.data, data.target)
x_train, x_valid, y_train, y_valid = train_test_split(x_ftrain, y_ftrain)

# Scaling X Data
x_train_scaled = scalar.fit_transform(x_train)
x_test_scaled = scalar.fit_transform(x_test)
x_valid_scaled = scalar.fit_transform(x_valid)

# Test, validation, and training data partitions
X_train_A, X_train_B = x_train_scaled[:, :5], x_train_scaled[:, 2:]
X_valid_A, X_valid_B = x_valid_scaled[:, :5], x_valid_scaled[:, 2:]
X_test_A, X_test_B = x_test_scaled[:, :5], x_test_scaled[:, 2:]
X_new_A, X_new_B = X_test_A[:3], X_test_B[:3]

# Visualization Network Statistics with Tensorboard
root_logdir = os.path.join(os.curdir, "/Users/ilyesouldsaada/Programming/Independent_Study_23-24/Neural_Networks/tensorboard_logs")

def get_run_logdir():
    # import time
    # run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S_test")
    return os.path.join(root_logdir, input("Name of test run? "))


# run_logdir = get_run_logdir()
#
# # Subclass WideAndDeep
# model = widedeepnet.WideAndDeep()
#
# model.compile(loss=["mse", "mse"], optimizer="sgd")
#
# # Tensorboard callback
# tensor_cb = keras.callbacks.TensorBoard(run_logdir)
# model.fit([X_train_A, X_train_B], [y_train, y_train],
#           epochs=20, validation_data=([X_valid_A, X_valid_B], [y_valid, y_valid]), callbacks=[tensor_cb])

# Wrapper object for CV

def create_model(n_hidden=1, n_neurons=50, learning_rate=3e-3, input_shape=[8]):
    model = keras.models.Sequential()
    options = {"input_shape": input_shape}
    model.add(keras.layers.InputLayer(shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
        options = {}
    model.add(keras.layers.Dense(1, **options))
    sgd_optimizer = keras.optimizers.SGD(learning_rate)
    model.compile(loss=['mse'], optimizer=sgd_optimizer)
    return model


# Create KerasRegressor object for optimization
wrapped_network = KerasRegressor(build_fn=create_model, n_hidden=1, n_neurons=50, learning_rate=3e-3, input_shape=[8])

# Gridsearch CV optimization
param_grid = {
    "n_hidden": [1,2,3,4],
    "n_neurons": np.arange(20,100,15),
    "learning_rate": reciprocal(3e-4, 3e-2)
}

rand_gridsearch = RandomizedSearchCV(wrapped_network, param_grid, cv=5)
rand_gridsearch.fit(x_train_scaled, y_train, epochs=75, validation_data=(x_valid_scaled,y_valid),
                    callbacks=[keras.callbacks.EarlyStopping(patience=3)])

# Displaying Results
print(f"Optimal score: {rand_gridsearch.best_score_}")
print(f"Optimal parameters: {rand_gridsearch.best_params_}")
print(f"Optimal model: {rand_gridsearch.best_estimator_}")

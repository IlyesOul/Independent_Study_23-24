import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


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

# Tinkering with weight initializers using Sequential feedforward network

model_uniform = tf.keras.models.Sequential([tf.keras.layers.Dense(50, "relu", kernel_initializer="he_normal"),
                                            tf.keras.layers.Dense(50, "relu", kernel_initializer="he_normal"),
                                            tf.keras.layers.Dense(1)])

model_uniform.compile(optimizer="sgd", metrics=['accuracy'], loss='mean_squared_error')

model_uniform.fit(x_train_scaled, y_train, validation_data=(x_valid_scaled, y_valid))

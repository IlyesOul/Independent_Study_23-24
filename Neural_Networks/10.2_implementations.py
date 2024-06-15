import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from widedeepnet import WideAndDeep
import statistics as stats


# Image classification network

# MNIST fashion dataset loading & preprocessing
# data = tf.keras.datasets.mnist
#
# (x_train, y_train), (x_test, y_test) = data.load_data()
#
# # Scaling
# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)
#
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten())
#
# # Sigmoid Activations Hidden
# model.add(tf.keras.layers.Dense(200, activation=tf.nn.sigmoid))
# model.add(tf.keras.layers.Dense(200, activation=tf.nn.sigmoid))
#
# # ReLu Activations hidden
# model.add(tf.keras.layers.Dense(200, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(200, activation=tf.nn.relu))
#
# # Output
# model.add(tf.keras.layers.Dense(200, activation=tf.nn.softmax))
#
# # Finalizing model
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# # Train model
# model.fit(x_train, y_train, epochs=3)
#
# # GENERALIZATION TESTING YEAH
# val_loss, val_accuracy = model.evaluate(x_test, y_test)
# print(f"Validation loss: {val_loss}")
# print(f"Validation accuracy: {val_accuracy}")


# Regression Network

scalar = MinMaxScaler()
data = fetch_california_housing()
#
# # Data splitting & preprocessing
x_ftrain, x_test, y_ftrain, y_test = train_test_split(data.data, data.target)
x_train, x_valid, y_train, y_valid = train_test_split(x_ftrain, y_ftrain)
#
# Scaling X Data
x_train_scaled = scalar.fit_transform(x_train)
x_test_scaled = scalar.fit_transform(x_test)
x_valid_scaled = scalar.fit_transform(x_valid)
#
# # Create Sequential Model
# model = tf.keras.models.Sequential()
#
# # Hidden Layers
# model.add(tf.keras.layers.Dense(100, "relu"))
# model.add(tf.keras.layers.Dense(100, "relu"))
# model.add(tf.keras.layers.Dense(100, "relu"))
#
# # Output layer
# model.add(tf.keras.layers.Dense(1))
#
# # Compile Model
# model.compile(optimizer='sgd',
#               loss='mean_squared_error',
#               metrics=['accuracy'])
#
# model.fit(x_train_scaled, y_train, epochs=20)
#
# # Evaluation of Model
# predictions = model.predict(x_test_scaled)
#
#
# print(f"MSE of Model is {mean_squared_error(predictions, y_test)}")
# val_loss, val_accuracy = model.evaluate(x_test, y_test)
# print(f"Validation loss: {val_loss}")
# print(f"Validation accuracy: {val_accuracy}")

# "Complex" Network #1 - Network has different paths and different kinds of inputs

# Input Layer
# input_A = tf.keras.layers.Input(shape=[5])
# input_B = tf.keras.layers.Input(shape=[6])
#
# # Hidden Layers
# hidden_1 = tf.keras.layers.Dense(75, "relu")(input_B)
# hidden_2 = tf.keras.layers.Dense(75, "relu")(hidden_1)
#
# # Concatenate Layer
# concat = tf.keras.layers.concatenate([input_A, hidden_2])
#
# # Output & Model Layers
# output = tf.keras.layers.Dense(1)(concat)
# model = tf.keras.models.Model(inputs=[input_A, input_B], outputs=[output])
#
# model.compile(loss="mse", optimizer="sgd", metrics=['accuracy'])
#
X_train_A, X_train_B = x_train_scaled[:, :5], x_train_scaled[:, 2:]
X_valid_A, X_valid_B = x_valid_scaled[:, :5], x_valid_scaled[:, 2:]
X_test_A, X_test_B = x_test_scaled[:, :5], x_test_scaled[:, 2:]
X_new_A, X_new_B = X_test_A[:3], X_test_B[:3]

# model.fit((X_train_A, X_train_B), y_train, epochs=15, validation_data=((X_valid_A, X_valid_B), y_valid))

# Complex Network #2 - Network retains the previous structure but can also have multiple kinds of output

# Input Layer
input_A = tf.keras.layers.Input(shape=[5])
input_B = tf.keras.layers.Input(shape=[6])

# Hidden Layers
hidden_1 = tf.keras.layers.Dense(200, activation="sigmoid")(input_B)
hidden_2 = tf.keras.layers.Dense(200, activation="sigmoid")(hidden_1)

# Concatenate Layer
concat = tf.keras.layers.concatenate([input_A, hidden_2])

# Outputs
output_A = tf.keras.layers.Dense(1)(concat)
output_B = tf.keras.layers.Dense(1)(hidden_2)

# Model creation & compilation
model = tf.keras.models.Model(inputs=[input_A, input_B], outputs=[output_A, output_B])
model.compile(loss=["mse", "mse"], optimizer="sgd", metrics=['accuracy', 'accuracy'])

# Model Fit
model.fit([X_train_A, X_train_B], [y_train, y_train], epochs=20, validation_data=([X_valid_A, X_valid_B], [y_valid, y_valid]))

# Model Results
predicted_values = model.predict([X_test_A, X_test_B])[0][:5]
print(f"Predicted values: {predicted_values}")
print(f"Actual values: {y_test}")

# Save Model
model.save("WideAndDeep.keras")

# Implementing a WideAndDeep Subclass object
# model = WideAndDeep()
#
# model.compile(loss=["mse", "mse"], optimizer="sgd")
# model.fit([X_train_A, X_train_B], [y_train, y_train], epochs=20, validation_data=([X_valid_A, X_valid_B], [y_valid, y_valid]))
#
# predicted_vals = model.predict([X_test_A, X_test_B])
# print(f"Test MSE: {mean_squared_error(predicted_vals[0], y_test)}")

# Create checkpoint callbacks for model during epoches
deployed_network = tf.keras.models.load_model("WideAndDeep.keras")

# Create checkpoint-callback and fit model with it
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("WideAndDeep.keras", save_best_only=True)
deployed_network.fit([X_train_A, X_train_B], y_train, validation_data=((X_valid_A,X_valid_B), y_valid), epochs=15, callbacks=[checkpoint_cb])

# Returns model that preforms best on valdiation due to "save_best_only"
model = tf.keras.load.load_model("WideAndDeep.keras")

# Early stopping
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
deployed_network.fit([X_train_A, X_train_B], y_train, validation_data=(([X_valid_A,X_valid_B]), y_valid), epochs=50, callbacks=[early_stopping_cb, checkpoint_cb])
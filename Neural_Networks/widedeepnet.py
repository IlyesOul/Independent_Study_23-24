import tensorflow as tf


# Wide-and-deep neural network class
class WideAndDeep(tf.keras.models.Model):

    # Initialization of WideAndDeep model parameters & layers
    def __init__(self, units=75, activation="sigmoid", **kwargs):
        super().__init__(**kwargs)
        self.hidden_1 = tf.keras.layers.Dense(units, activation)
        self.hidden_2 = tf.keras.layers.Dense(units, activation)
        self.main_output = tf.keras.layers.Dense(1)
        self.aux_output = tf.keras.layers.Dense(1)

    # Method that effectively constructs the network upon invocation
    def call(self, inputs):
        input_a, input_b = inputs
        hidden_1 = self.hidden_1(input_b)
        hidden_2 = self.hidden_2(hidden_1)
        concat = tf.keras.layers.concatenate([hidden_1, input_a])
        main_output = self.main_output(concat)
        aux_output = self.aux_output(hidden_2)

        return main_output, aux_output


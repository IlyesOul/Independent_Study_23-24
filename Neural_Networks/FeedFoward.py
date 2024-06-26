import tensorflow as tf

# Simple Feedforward network class


class FeedForward(tf.keras.models.Model):

    # Constructor
    def __init__(self, units=125, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.hidden_1 = tf.keras.layers.Dense(units, activation)
        self.hidden_2 = tf.keras.layers.Dense(units, activation)
        self.output = tf.keras.layers.Dense(1)


    # Call method
    def call(self, input):
        hidden_1 = self.hidden_1(input)
        hidden_2 = self.hidden_2(hidden_1)
        output = self.output(hidden_2)

        return output
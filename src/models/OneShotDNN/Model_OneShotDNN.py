import tensorflow as tf

# Model definition template

class OneShotDNN(tf.keras.Model):

  def __init__(self, INPUT_DIM, OUTPUT_DIM, **kwargs):
    
    super().__init__()
    # Intialize layers
    self.dense1 = tf.keras.layers.Dense(128, activation=tf.nn.relu, input_shape=(INPUT_DIM,))
    self.dense2 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
    self.dense3 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
    self.dense4 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
    self.dense_out = tf.keras.layers.Dense(OUTPUT_DIM)

  def call(self, inputs, training=False):
    # Define forward pass
    x = self.dense1(inputs)
    x = self.dense2(x)
    x = self.dense3(x)
    x = self.dense4(x)
    return self.dense_out(x)



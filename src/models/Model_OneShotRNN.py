import tensorflow as tf

# Model definition template

class OneShotRNN(tf.keras.Model):

  def __init__(self, INPUT_DIM, OUTPUT_DIM, **kwargs):
    
    super().__init__()
    # Intialize layers
    rnn_cells = [tf.keras.layers.LSTMCell(128) for _ in range(2)]
    stacked_lstm = tf.keras.layers.StackedRNNCells(rnn_cells)
    self.lstm_layer = tf.keras.layers.RNN(stacked_lstm)
    self.dense_out = tf.keras.layers.Dense(OUTPUT_DIM)


  def call(self, inputs, training=False):
    # Define forward pass
    x = self.lstm_layer(inputs)
    return self.dense_out(x)
    



from tensorflow.keras.layers import InputLayer, Dense, LSTM
from tensorflow.keras import Sequential

def Build_LSTM_Model(input_dim, pred_dim, skip, units, d_acti, **kwargs):

    inp_d = int(input_dim / skip)
    pred_d = int(pred_dim / skip)

    layers = [InputLayer(input_shape=(inp_d, 4))]
    for i in range(len(units)-2):
        layers.append(LSTM(units[i], return_sequences=True))
    layers.append(LSTM(units[-2], return_sequences=False))
    layers.append(Dense(units[-1], activation=d_acti))
    layers.append(Dense(pred_d))

    model = Sequential(layers)

    return model

def Build_Base_Model(input_dim, pred_dim, skip, **kwargs):

    inp_d = int(input_dim / skip)
    pred_d = int(pred_dim / skip)

    layers = [InputLayer(input_shape=(inp_d, 4))]
    layers.append(LSTM(32, return_sequences=False))
    layers.append(Dense(128))
    layers.append(Dense(pred_d))

    model = Sequential(layers)

    return model
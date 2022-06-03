from src.data.Dataset import Dataset as DS
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os

# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")   # Run only if necessary



def Train_func(**para):
    
    Data        = DS(**para)

    INPUT_DIM = Data.xshape
    OUTPUT_DIM = Data.yshape

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=INPUT_DIM),
        tf.keras.layers.LSTM(units=256, return_sequences=True),
        tf.keras.layers.LSTM(units=64, return_sequences=False),
        tf.keras.layers.Dense(64, activation='linear'),
        tf.keras.layers.Dense(OUTPUT_DIM[0])
    ])
    
    def compile_model(optimizer, loss, metrics, **kwargs):
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        model.summary()

    # Model compile
    compile_model(**para)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=7,
                                                    mode='min')

    checkpoint_filepath = 'models/RNN/14/model'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                patience=5, min_lr=0.000001)

    logdir = "logs/14/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, update_freq=500)
    
    # Model fit
    train_history = model.fit(
        Data.Val,
        epochs=para['epochs'],
        verbose=1,
        validation_data=Data.Test,
        callbacks=[
            model_checkpoint_callback,
            tensorboard_callback
        ]
    )

    return train_history ,model

def load_parameters(path):
    with open(path, 'r') as j:
        contents = json.loads(j.read())
    return contents

def ARModel_Inference(x, y, OUT_dim, model):
    inp1 = x
    true_roll = y[:,:,0]
    pred_roll = []
    for i in range(OUT_dim):
        wave = y[:,i:i+1,-1:]
        INPUT = inp1
        OUT = model(INPUT)
        roll = tf.expand_dims(tf.cast(OUT, tf.float64), axis=1)
        HPW = y[:,i:i+1,1:]
        temp = tf.concat([roll, HPW], axis = -1)
        inp1 = tf.concat([inp1[:,1:,:], temp], axis=1)
        pred_roll.append(roll[:,:,0])

    pred_roll = tf.squeeze(tf.stack(pred_roll, axis=1), [-1]) 
    return true_roll, pred_roll


def get_inference(Data_inf, model, OUT_dim, save_dir=None):
    true_roll = []
    pred_roll = []
    inputs = []
    for x, y in Data_inf:
        t_r, p_r = ARModel_Inference(x, y, OUT_dim = OUT_dim, model=model)
        pred_roll.extend(p_r)
        true_roll.extend(t_r)
        inputs.extend(x)
        
    inputs = np.array(inputs)
    true_roll = np.array(true_roll)
    pred_roll = np.array(pred_roll)
        
    return inputs, true_roll, pred_roll
    
def save_inference(inputs, true_roll, pred_roll, save_dir):
    
    
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        
    inp_path = save_dir + '/inputs.csv'
    true_path = save_dir + '/true_roll.csv'
    pred_path = save_dir + '/pred_roll.csv'

    np.savetxt(inp_path, inputs[:,:,0], delimiter =", ")
    np.savetxt(true_path, true_roll, delimiter =", ")
    np.savetxt(pred_path, pred_roll, delimiter =", ")
    
def load_inference(folder):
    inputs    = np.genfromtxt(folder + '/inputs.csv', delimiter=', ', skip_header=0)
    true_roll = np.genfromtxt(folder + '/true_roll.csv', delimiter=', ', skip_header=0)
    pred_roll = np.genfromtxt(folder + '/pred_roll.csv', delimiter=', ', skip_header=0)
    
    return inputs, true_roll, pred_roll



path = 'parameters.json'
val_dir = 'models/RNN/11/Val'
test_dir = 'models/RNN/11/Test'
model_lastepoch_path = 'models/RNN/14/model_lastepoch'
model_best_path = 'models/RNN/14/model'

contents = load_parameters(path)

history, model = Train_func(**contents)

model.save(model_lastepoch_path)

# Load best model
# model = tf.keras.models.load_model(model_best_path)


# # Run inference

# test_in, test_true, test_pred = get_inference(Data_inf.Test, save_dir=test_dir)
# val_in, val_true, val_pred = get_inference(Data_inf.Val, save_dir=val_dir)

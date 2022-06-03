import tensorboard
from src.data.Dataset import Dataset as DS
# from src.callbacks.epochcheckpoint import EpochCheckpoint
# from src.callbacks.trainingmonitor import TrainingMonitor
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os
from pathlib import Path

PATH_TO_CONFIG_FILE = Path('parameters.json')
PATH_TO_MODEL_DIR = Path('models') / 'AQUA_RUNS'
MODEL_NUMBER = '01'
# Read parameters.json

with open(PATH_TO_CONFIG_FILE) as json_file:
    para = json.load(json_file)

# Define Dataset

Train_data = DS(**para)

INPUT_DIM = Train_data.xshape
OUTPUT_DIM = Train_data.yshape

# Define model

model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=INPUT_DIM),
        tf.keras.layers.LSTM(units=256, return_sequences=True),
        tf.keras.layers.LSTM(units=64, return_sequences=False),
        tf.keras.layers.Dense(64, activation='linear'),
        tf.keras.layers.Dense(OUTPUT_DIM[0])
    ])

# Compile model
def Compile_model(loss, metrics, lr, b1, b2, epsi, amsgrad, **kwargs):
    opti = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=b1, beta_2=b2, epsilon=epsi, amsgrad=amsgrad)
    model.compile(optimizer=opti, loss=loss, metrics=metrics)
    model.summary()

Compile_model(**para)

# Define callbacks

model_ckpt = tf.keras.callbacks.ModelCheckpoint(
    filepath=PATH_TO_MODEL_DIR / MODEL_NUMBER /'model',
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5,
    patience=10,
    min_lr=0.00001)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=PATH_TO_MODEL_DIR / MODEL_NUMBER /'logs',
    update_freq=500)

backup = tf.keras.callbacks.BackupAndRestore(
    filepath=PATH_TO_MODEL_DIR / MODEL_NUMBER /'backup')

logger = tf.keras.callbacks.CSVLogger(
    PATH_TO_MODEL_DIR / MODEL_NUMBER /'Training_log/log.csv')


callbacks = [model_ckpt, backup, reduce_lr, logger]

# Training

history = model.fit(
    Train_data.Val,
    epochs=para['epochs'],
    verbose=2,
    validation_data=Train_data.Test,
    callbacks=callbacks)

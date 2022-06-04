import time

s = time.time()
import tensorboard
from src.data.Dataset import Dataset as DS
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os
from pathlib import Path
from BuildModel import Build_LSTM_Model, Build_Base_Model
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-r", "--rundir", default="AQUA_RUNS", help="Run directory")
ap.add_argument("-n", "--modelnum", type=str, help="Model number")
args = vars(ap.parse_args())

RUN_NAME = str(args['rundir'])
MODEL_NUMBER = str(args['modelnum'])
PATH_TO_MODEL_DIR = Path('models') / RUN_NAME / MODEL_NUMBER
PATH_TO_CONFIG_FILE = PATH_TO_MODEL_DIR / 'parameters.json'

logger_path = PATH_TO_MODEL_DIR / 'Training_log' 

if not os.path.isdir(logger_path):
    os.makedirs(logger_path)

# Read parameters.json

with open(PATH_TO_CONFIG_FILE) as json_file:
    para = json.load(json_file)

print("SETUP TIME = ", time.time() - s)

s = time.time()

# Define Dataset
Train_data = DS(**para)

print("DATASET TIME = ", time.time() - s)

s = time.time()

# Define model
if para['model_type'] == 'LSTM':
    model = Build_LSTM_Model(**para)
elif para['model_type'] == 'Base':
    model = Build_Base_Model(**para)

# Compile model
def Compile_model(loss, metrics, lr, b1, b2, epsi, amsgrad, **kwargs):
    opti = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=b1, beta_2=b2, epsilon=epsi, amsgrad=amsgrad)
    model.compile(optimizer=opti, loss=loss, metrics=metrics)
    model.summary()

Compile_model(**para)

print("MODEL COMPILE TIME = ", time.time() - s)

# Define callbacks

model_ckpt = tf.keras.callbacks.ModelCheckpoint(
    filepath=PATH_TO_MODEL_DIR / 'model',
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=para['red_lr_f'],
    patience=para['red_lr_patience'],
    min_lr=para['red_lr_min_lr'])

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=PATH_TO_MODEL_DIR / 'logs',
    update_freq=500)

backup = tf.keras.callbacks.BackupAndRestore(
    PATH_TO_MODEL_DIR / 'backup')

logger = tf.keras.callbacks.CSVLogger(
    logger_path / 'log.csv', append=True)


callbacks = [model_ckpt, backup, reduce_lr, logger]

# Training

history = model.fit(
    Train_data.Val,
    epochs=para['epochs'],
    verbose=2,
    validation_data=Train_data.Test,
    callbacks=callbacks,
    batch_size=para['batch_size'])

import time

s = time.time()

from src.data.Dataset import Dataset as DS
import tensorflow as tf
import json
from pathlib import Path
import argparse
from InferenceUtils import *

ap = argparse.ArgumentParser()
ap.add_argument("-r", "--rundir", default="AQUA_RUNS", help="Run directory")
ap.add_argument("-n", "--modelnum", type=str, help="Model number")
args = vars(ap.parse_args())

RUN_NAME = str(args['rundir'])
MODEL_NUMBER = str(args['modelnum'])
PATH_TO_MODEL_DIR = Path('models') / RUN_NAME / MODEL_NUMBER
PATH_TO_CONFIG_FILE = PATH_TO_MODEL_DIR / 'inference.json'

# Read parameters.json

with open(PATH_TO_CONFIG_FILE) as json_file:
    para = json.load(json_file)

print("SETUP TIME = ", time.time() - s)

s = time.time()
# Load model

model_dir = PATH_TO_MODEL_DIR / "model"

model = tf.keras.models.load_model(model_dir)

print("MODEL LOAD TIME = ", time.time() - s)

s = time.time()

# Define Dataset
Data_inf = DS(**para)

print("DATASET TIME = ", time.time() - s)

val_dir = PATH_TO_MODEL_DIR / "val"
test_dir = PATH_TO_MODEL_DIR / "test"

OUT_dim = para['output_dim']
batch_size = para['batch_size']

val_in, val_true, val_pred, val_inf_t = get_inference(Data_inf.Val, model, OUT_dim)

print(f"Average Inference time for Validation set with batch size {batch_size} and out_dim {OUT_dim}  = {val_inf_t}")
# save_inference(val_in, val_true, val_pred, val_dir)

test_in, test_true, test_pred, test_inf_time = get_inference(Data_inf.Test, model, OUT_dim)

print(f"Average Inference time for Test set with batch size {batch_size} and out_dim {OUT_dim}  = {test_inf_time}")
# save_inference(test_in, test_true, test_pred, test_dir)
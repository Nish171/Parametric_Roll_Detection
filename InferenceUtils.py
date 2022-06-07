import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

def ARModel_Inference(x, y, OUT_dim, model):
    x, y = tf.cast(x, tf.float32), tf.cast(y, tf.float32)
    inp1 = x
    true_roll = y[:,:,0]
    pred_roll = []
    s = time.time()
    for i in range(OUT_dim):
        # wave = y[:,i:i+1,-1:]
        INPUT = inp1
        OUT = model(INPUT)
        roll = tf.expand_dims(tf.cast(OUT, tf.float32), axis=1)
        HPW = y[:,i:i+1,1:]
        temp = tf.concat([roll, HPW], axis = -1)
        inp1 = tf.concat([inp1[:,1:,:], temp], axis=1)
        pred_roll.append(roll[:,:,0])
    e = time.time()
    pred_roll = tf.squeeze(tf.stack(pred_roll, axis=1), [-1]) 
    return true_roll, pred_roll, e-s


def get_inference(Data_inf, model, OUT_dim, inf_time_dir=None):
    true_roll = []
    pred_roll = []
    inputs = []
    # batch = []
    ti = []
    total_t = 0
    i=1
    for x, y in Data_inf.take(2):
        t_r, p_r, t = ARModel_Inference(x, y, OUT_dim = OUT_dim, model=model)
        pred_roll.extend(p_r)
        true_roll.extend(t_r)
        inputs.extend(x)
        # batch.extend(i)
        ti.append(t)
        total_t += t
        i+=1
    if inf_time_dir:
        np.savetxt(inf_time_dir / 'inf_time.csv', ti, delimiter=', ')
        
    inputs = np.array(inputs)
    true_roll = np.array(true_roll)
    pred_roll = np.array(pred_roll)
        
    return inputs, true_roll, pred_roll, total_t/i
    
def save_inference(inputs, true_roll, pred_roll, save_dir):
    
    
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        
    inp_path = save_dir / 'inputs.csv'
    true_path = save_dir / 'true_roll.csv'
    pred_path = save_dir / 'pred_roll.csv'

    np.savetxt(inp_path, inputs[:,:,0], delimiter =", ")
    np.savetxt(true_path, true_roll, delimiter =", ")
    np.savetxt(pred_path, pred_roll, delimiter =", ")
    
def load_inference(folder):
    inputs    = np.genfromtxt(folder / 'inputs.csv', delimiter=', ', skip_header=0)
    true_roll = np.genfromtxt(folder / 'true_roll.csv', delimiter=', ', skip_header=0)
    pred_roll = np.genfromtxt(folder / 'pred_roll.csv', delimiter=', ', skip_header=0)
    
    return inputs, true_roll, pred_roll

def plot_inference(inputs, true, pred, nos, cut=0.5, units='deg'):
    n = inputs.shape[-1]
    n_plots = len(nos)
    cut_ind = int(n*cut)
    
    fac = 180/np.pi if units=='deg' else 1
    
    plt.figure(figsize=(18, n_plots*5))
    
    w1_end = inputs.shape[1]
    w2_size = true.shape[-1]*cut_ind
    t1 = np.array(range(0, w1_end))*0.25
    t2 = np.array(range(w1_end, w1_end + cut_ind))*0.25
    
    for i, ind in enumerate(nos):
        plt.subplot(n_plots, 1, i+1)
        plt.plot(t1, inputs[ind]*fac, label='Input')
        plt.plot(t2, true[ind,:cut_ind]*fac, label='True_roll')
        plt.plot(t2, pred[ind,:cut_ind]*fac, label='Pred_roll')
        plt.ylabel(f"Roll angle ({units})")
        plt.legend()
    plt.xlabel("Time (s)")    
    plt.show()
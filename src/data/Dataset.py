import wave
import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import math

class Dataset:
    def __init__(self, input_dim, pred_dim, shift, train_s=0.8, val_s=0.1, roll_period=30, skip=1, hop=0.25, batch_size=1, classification=False, in_cols=['roll'], out_cols=['roll'], **kwargs):
        self.data_folder                    = Path("data")
        self.sim_folder_name                = "Simulations_01"
        self.num_sims                       = 62
        self.raw_data_folder                = self.data_folder / 'raw'
        self.processed_data_folder          = self.data_folder / 'processed'
        self.interim_data_folder            = self.data_folder / 'interim'
        self.sim_data_folder                = self.processed_data_folder / self.sim_folder_name
        self.roll_period                    = roll_period                                # Estimated roll period in seconds
        self.sr                             = 0.25                              # Sampling rate of simulation data in seconds
        self.roll_thres                     = 10 * np.pi / 180                               # Threshold for roll angle
        self.input_dim                      = input_dim                         # Number of seconds of data to be used as input
        self.pred_dim                       = pred_dim                          # Number of seconds of data to be predicted
        self.shift                          = shift                             # Number of seconds to shift the prediction window
        self.skip                           = int(skip/self.sr)                 # Interval of the required data
        self.hop                            = hop                               # fraction of hop size in terms of input window size
        self.xshape                         = [int(math.ceil(self.input_dim/self.skip/self.sr)), len(in_cols)]
        self.yshape                         = [int(math.ceil(self.pred_dim/self.skip/self.sr)), len(out_cols)]
        self.train_split                    = train_s
        self.val_split                      = val_s
        self.rs                             = 11
        self.batch_size                     = batch_size
        self.train, self.test, self.val     = self.train_test_val(self.train_split, self.val_split, self.rs)
        self.classification                 = classification
        self.in_cols                        = in_cols
        self.out_cols                       = out_cols

        if self.classification:
            self.yshape = [1, len(self.out_cols)] 



    def get_sim_data(self, sim_num, split=True):
        '''
        Input: 
            sim_num: Simulation number
        
        Returns: 
            roll: Roll angle of the ship in radians
            pitch: Pitch angle of the ship in radians
            heave/L: Heave of the ship normalized to L
            wave height/L: Wave height of the ship normalized to L

        '''
        inp_path    = self.sim_data_folder / 'sim_{}'.format(sim_num) / 'motion.csv'
        wave_path   = self.sim_data_folder / 'sim_{}'.format(sim_num) / 'wave.csv'
        data        = pd.read_csv(inp_path)
        wave_data   = pd.read_csv(wave_path)
        w_cols      = wave_data.columns
        cols        = data.columns
        time        = data[cols[0]]
        heave       = data[cols[9]] / 230
        roll        = data[cols[10]] * np.pi / 180
        pitch       = data[cols[11]] * np.pi / 180
        wave        = wave_data[w_cols[1]] / 230
        if split:
            return time, heave, roll, pitch, wave
        # data2 = data.iloc[:, [0, 9, 10, 11]]
        data2 = pd.concat([time, heave, roll, pitch, wave], axis=1)
        data2.columns = ['time', 'heave', 'roll', 'pitch', 'wave']
        return data2

    def get_sim_inputs(self, sim_num):
        '''
        Function: Get the wave height and wave period used to perform the simulation

        Input:
            sim_num: Simulation number

        Returns:
            Hs: Wave height
            Tp: Wave period
        '''
        inp_path    = self.sim_data_folder / 'sim_{}'.format(sim_num) / 'KCS.txt'
        
        with open(inp_path, 'r') as file:
            data    = file.readlines()
            
        line        = data[72].split(' ')
        Hs          = line[0]
        Tp          = line[1]
        
        return Hs, Tp

    def _data_stats_(self):
        '''
        Function: Get the statistics of the data
        '''
        stats               = pd.DataFrame(columns=['Sim_no', 'Hs', 'Tp', 'Max_roll'])
        
        for i in range(1, self.num_sims+1):
            _, _, roll, _, _   = self.get_sim_data(i)
            max_roll        = max(roll)
            Hs, Tp          = self.get_sim_inputs(i)
            stats           = stats.append({'Sim_no': i, 'Hs':Hs, 'Tp':Tp, 'Max_roll':max_roll}, ignore_index=True)
        return stats

    def stats_table(self):
        '''
        Function: Get the statistics of the data

        Returns:
            stats: DataFrame containing the statistics of the data
        
        '''
        stats = self._data_stats_()
        cols = stats['Tp'].unique()
        rows = stats['Hs'].unique()
        stats_df = pd.DataFrame(index=rows, columns=cols)
        for tp in cols:
            for hs in rows:
                hs_df = stats[stats['Hs']==hs]
                tp_df = hs_df[hs_df['Tp']==tp]
                if len(tp_df)>0:
                    stats_df[tp][hs] = float(tp_df['Max_roll'])
                    
        return stats_df  

    def train_test_val(self, train_split=0.8, val_split=0.1, rs = 11):
        '''
        Function: Split the data into train, test and validation sets

        Input:
            train_split: Fraction of the data to be used for training
            val_split: Fraction of the data to be used for validation
            rs: Random state

        Returns:
            train: Numpy array containing the training data sim nos
            test: Numpy array containing the test data sim nos
            val: Numpy array containing the validation data sim nos
        
        '''
        stats       = self._data_stats_()
        para        = np.array(stats[stats['Max_roll']>self.roll_thres]['Sim_no'])
        non_para    = np.array(stats[stats['Max_roll']<=self.roll_thres]['Sim_no'])
        
        para_train, para_rem            = train_test_split(para, train_size=self.train_split, random_state=rs)
        non_para_train, non_para_rem    = train_test_split(non_para, train_size=self.train_split, random_state=rs)

        val_split = val_split/(1-train_split)
        
        para_val, para_test             = train_test_split(para_rem, train_size=self.val_split, random_state=rs)
        non_para_val, non_para_test     = train_test_split(non_para_rem, train_size=self.val_split, random_state=rs)
        
        train   = np.concatenate((para_train, non_para_train))
        test    = np.concatenate((para_test, non_para_test))
        val     = np.concatenate((para_val, non_para_val))
        return train, test, val 

    

    def window_gen(self, sim_nos):
        '''
        Function: Generate the windows for the data

        Input:
            sim_nos: Numpy array containing the simulation numbers
        Returns:
            Generates a generator object that yields the windows

        '''
        def callable_gen():
            '''
            Function: Generator function

            Returns:
                Yields the windows
                X: Numpy array containing the data. Shape: (Input_window_size, Input_num_features)
                Y: Numpy array containing the labels. Shape: (Output_window_size, Output_num_features)
            '''
            for sim_no in sim_nos:

                data            = self.get_sim_data(sim_no, split=False)
                data            = data.loc[:, list(set(self.in_cols+self.out_cols))]
                sig_len         = len(data)
                win_size        = int(self.input_dim/self.sr)
                pred_win_size   = int(self.pred_dim/self.sr)
                shift_size      = int(self.shift/self.sr)
                total_win_size  = win_size + shift_size
                hop_len         = int(self.hop/self.sr)
                num             = int(1 + (sig_len - total_win_size) // hop_len)
                
                for i in range(num):
                    
                    win_start       = (i*hop_len)
                    win_end         = win_size + win_start
                    pred_win_end    = win_end + shift_size
                    pred_win_start  = pred_win_end - pred_win_size
                    x               = data.iloc[win_start:win_end:self.skip,:].loc[:, self.in_cols]
                    y               = data.iloc[pred_win_start:pred_win_end:self.skip,:].loc[:, self.out_cols]
                    if self.classification:
                        y = [int(max(y) > self.roll_thres)]
                        # if max(y) > self.roll_thres:
                        #     y = [1]
                        # else:
                        #     y = [0]
                    yield np.array(x), np.array(y)

        return callable_gen

    def make_tf_dataset(self, sim_nos):
        '''
        Function: Generate the TF dataset

        Input:
            sim_nos: Numpy array containing the simulation numbers

        Returns:
            tf_dataset: TF dataset
        '''
        ds = self.window_gen(sim_nos)
        dataset = tf.data.Dataset.from_generator(
        ds, 
        output_signature=
        (tf.TensorSpec(shape=self.xshape, dtype=tf.float64), 
        tf.TensorSpec(shape=self.yshape, dtype=tf.float64)))

        return dataset.batch(self.batch_size)

    def normalizer(self, norm):
        for x, y in self.Train:
            norm.adapt(x)
        return norm

    @property
    def Train(self):
        '''
        Function: Get the training data

        Returns:
            tf_dataset: TF dataset containing the training data
        '''
        return self.make_tf_dataset(self.train)

    @property
    def Test(self):
        '''
        Function: Get the test data

        Returns:
            tf_dataset: TF dataset containing the test data
        '''
        return self.make_tf_dataset(self.test)

    @property
    def Val(self):
        '''
        Function: Get the validation data

        Returns:
            tf_dataset: TF dataset containing the validation data
        '''
        return self.make_tf_dataset(self.val)

    def Example(self, sim_no):
        return self.make_tf_dataset([sim_no])

    def plot_example(self, sim_no, model=None, max_plots=3, classification=False):
        i=0
        plt.figure(figsize=(12,max_plots*4))
        for x, y in self.Example(sim_no):
            if max(y[0,:])>self.roll_thres:
                i+=1
                delta = int(self.skip*self.sr)
                in_win_s = int(0/self.sr)
                in_win_e = int(self.input_dim/self.sr)
                pred_win_s = int((self.input_dim + self.shift - self.pred_dim)/self.sr)
                pred_win_e = pred_win_s + int(self.pred_dim/self.sr)
                t1 = np.array(range(in_win_s, in_win_e))*self.sr
                t2 = np.array(range(pred_win_s, pred_win_e))*self.sr
                plt.subplot(max_plots, 1, i)
                plt.plot(t1, x[0, :, 0], label='Input')
                if (model is not None):
                    y_pred = model.predict(x)
                    if classification:
                        text = 'Predicted: ' + str('False' if y_pred[0,0]<0.3 else 'True')
                        plt.text(0, 0, text, fontsize=12)
                    else:
                        plt.plot(t2, y_pred[0,:], label='Prediction')
                        text1 = "MSE: " + str(tf.keras.metrics.MeanSquaredError()(y[0,:], y_pred[0,:]).numpy())
                        text2 = "MAE: " + str(tf.keras.metrics.MeanAbsoluteError()(y[0,:], y_pred[0,:]).numpy())
                        plt.text(50, max(y[0,:])-2, text1, fontsize=12)
                        plt.text(150, max(y[0,:])-2, text2, fontsize=12)
                plt.plot(t2, y[0,:], label='Actual')
                plt.ylabel("Roll angle (deg)")
            if i==1:
                plt.legend()
            if i==max_plots:
                plt.xlabel("Time (s)")
                plt.show()
                break
                
    def plot_sim(self, sim_no):
        time, heave, roll, pitch, wave = self.get_sim_data(sim_no)

        fig, ax = plt.subplots(4)
        fig.set_figheight(12)
        fig.set_figwidth(12)

        ax[0].plot(time, heave)
        ax[0].set_ylabel("Heave / L", fontsize=15)

        ax[1].plot(time, roll)
        ax[1].set_ylabel("Roll (Rad)", fontsize=15)

        ax[2].plot(time, pitch)
        ax[2].set_ylabel("Pitch (Rad)", fontsize=15)

        ax[3].plot(time, wave)
        ax[3].set_ylabel("Wave height / L ", fontsize=15)

        fig.tight_layout()
        plt.xlabel("Time (sec)", fontsize=15)
        plt.show()

    def div_windows(self, sim_nos, n, m, hop=0.25):
        
        x = []
        y = []

        for sim_no in sim_nos:

            _, _, roll, _ = self.get_sim_data(sim_no)
            roll = np.array(roll)
            sig_len = len(roll)
            win_size = int(n*self.roll_period/self.sr)
            hop_len = int(win_size*hop)
            num = int(1 + (sig_len - win_size) // hop_len)
            pred_win_size = int(m*self.roll_period/self.sr)
            
            
            for i in range(num):
                
                win_start = (i*hop_len)
                win_end = win_size + win_start
                pred_win_end = win_end + pred_win_size
                if pred_win_end < sig_len:
                    x.append(roll[win_start:win_end])
                    y.append(roll[win_end:pred_win_end])
                    
        return np.array(x), np.array(y)

    def make_csv_split(self, data, n, m, split="train"):
        X, Y = self.div_windows(data, n, m)

        pro_data_fld = os.path.join(self.processed_data_folder, self.sim_folder_name)
        n_m_fld = os.path.join(pro_data_fld, 'rp_{}_n_{}_m_{}'.format(self.roll_period ,n, m))
        split_data_fld = os.path.join(n_m_fld, '{}'.format(split))

        if not os.path.isdir(pro_data_fld):
            os.mkdir(pro_data_fld)
        
        if not os.path.isdir(n_m_fld):
            os.mkdir(n_m_fld)
        
        if not os.path.isdir(split_data_fld):
            os.mkdir(split_data_fld)

        np.savetxt(os.path.join(split_data_fld, '{}_X.csv'.format(split)), X, delimiter=',')
        np.savetxt(os.path.join(split_data_fld, '{}_Y.csv'.format(split)), Y, delimiter=',')
            
    def make_csv_files(self, n, m, train_split=0.8, val_split=0.1, rs = 11):
        train, test, val = self.train_test_val(train_split, val_split, rs)
        self.make_csv_split(train, n, m, split="train")
        self.make_csv_split(test, n, m, split="test")
        self.make_csv_split(val, n, m, split="val")    






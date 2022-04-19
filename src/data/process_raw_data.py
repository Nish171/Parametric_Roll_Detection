import argparse
import numpy as np
import pandas as pd
import os
import glob
import shutil
from pathlib import Path

def copy_text_files(src_dir, dst_dir):
    with open(src_dir,'r') as firstfile, open(dst_dir,'w') as secondfile:
    # read content from first file
        for line in firstfile:  
                # write content to second file
                secondfile.write(line)

def copy_csv_files(src_dir, dst_dir):
    df = pd.read_csv(src_dir)
    df.to_csv(dst_dir, index=False)


parser = argparse.ArgumentParser(description='Process raw data')
parser.add_argument('-i', '--input', type=str, default='Simulations_01')
parser.add_argument('-o', '--output', type=str, default='Simulations_01')

args = parser.parse_args()

sim_folder      = args.input
output_folder   = args.output

data = Path('data')
raw_data_folder         = data / 'raw' / sim_folder
processed_data_folder   = data / 'processed' / output_folder

print('Processing raw data from: {}'.format(raw_data_folder))
print('Saving processed data to: {}'.format(processed_data_folder))
try:
    os.makedirs(processed_data_folder)
except:
    pass

if os.path.isdir(raw_data_folder):
    sim_fols = glob.glob(str(raw_data_folder / '*'))
    print(len(sim_fols))
    for sim_fol in sim_fols:
        sim_name        = os.path.basename(sim_fol)
        sim_fol = Path(sim_fol)
        dst = processed_data_folder / sim_name
        try:
            os.makedirs(dst)
        except:
            pass
        motion_data_src = sim_fol / 'motion' / 'KCS_MotionTimeHistory.csv'
        motion_data_dst = processed_data_folder / sim_name / 'motion.csv'

        wave_data_src = sim_fol / 'wave' / 'KCS_WaveTimeHistory.csv'
        wave_data_dst = processed_data_folder / sim_name / 'wave.csv'

        input_data_src  = sim_fol / 'KCS.inp'
        input_data_dst  = processed_data_folder / sim_name / 'KCS.txt'
        copy_text_files(input_data_src, input_data_dst)
        copy_csv_files(motion_data_src, motion_data_dst)
        copy_csv_files(wave_data_src, wave_data_dst)
        
    print("Raw data has been processed and saved to {}".format(processed_data_folder))
else:
    print('{} is not a valid folder'.format(raw_data_folder))






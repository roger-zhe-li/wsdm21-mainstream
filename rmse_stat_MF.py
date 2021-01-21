import os
import sys
import numpy as np 
import json
import pandas as pd 
import argparse

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

parser = argparse.ArgumentParser(description='parameters for RMSE computation.')
parser.add_argument('--dataset', type=str, default='music_instruments',
					choices=['music_instruments', 'instant_video', 'digital_music', 'beer'])
parser.add_argument('--data_store_path', nargs='?', default='./data/',
					help='Input data path.')
parser.add_argument('--num_latent_factor', type=int, default=100,
					help='latent factor number')
parser.add_argument('--flush', action='store_true', default=False,
					help='enables data rebuilding')
args = parser.parse_args()

data_store_path = args.data_store_path
dataset = args.dataset

seeds = [1, 777, 1234, 1476, 1573, 1771, 1842, 1992, 2003, 2020]

rmse_dir = os.path.join(data_store_path, dataset)

txt_path = os.path.join(data_store_path, 'rmse.txt')

if args.flush:
	os.remove(txt_path)

with open(txt_path, 'a+') as f:
	f.write('\n'+dataset+'\n\n')

rmse_score = []
for seed in seeds:
	rmse_path = os.path.join(rmse_dir, 'MF_'+str(seed), 'latent_factor_'+str(args.num_latent_factor), 'mode_5', 'test_result.json')
	with open(rmse_path, 'r') as f:
		info = json.load(f)
		rmse = info['rmse']
		rmse_score.append(rmse)
avr = np.mean(rmse_score)
std = np.std(rmse_score, ddof = 1)
with open(txt_path, 'a+') as f:
	f.write('avr=%5.4f, std=%5.4f\n'% (avr, std))
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
weights_u = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
weights_i = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

rmse_dir = os.path.join(data_store_path, dataset)

txt_path = os.path.join(data_store_path, 'rmse.txt')

if args.flush:
	os.remove(txt_path)

with open(txt_path, 'a+') as f:
	f.write('\n'+dataset+'\n\n')

for weight_u in weights_u:
	for weight_i in weights_i:
		rmse_score = []
		if weight_u == 0 or weight_i == 0 or weight_u == weight_i:
			for seed in seeds:
				rmse_path = os.path.join(rmse_dir, 'NAECF_'+str(seed), 'latent_factor_'+str(args.num_latent_factor), 'mode_4', 'user_'+str(weight_u)+'_item_'+str(weight_i), 'test_result.json')
				with open(rmse_path, 'r') as f:
					info = json.load(f)
					rmse = info['rmse']
					rmse_score.append(rmse)
			avr = np.mean(rmse_score)
			std = np.std(rmse_score, ddof = 1)
			with open(txt_path, 'a+') as f:
				f.write('weight_u=%2.1f, weight_i=%2.1f, avr=%5.4f, std=%5.4f\n'% (weight_u, weight_i, avr, std))


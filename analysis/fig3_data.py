# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from pygam import LinearGAM, s, f, GammaGAM, LogisticGAM
import pandas as pd
import os
import sys
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse
import warnings
from scipy.stats import describe
from collections import Counter
warnings.filterwarnings("ignore")
matplotlib.use('Agg')
np.set_printoptions(suppress=True)
plt.rcParams["figure.figsize"] = (5, 4)
plt.rc('xtick',labelsize=12,)
plt.rc('ytick',labelsize=12, )
# plt.yticks(fontproperties = 'Times New Roman', size = 5)
# plt.xticks(fontproperties = 'Times New Roman', size = 5)



dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

parser = argparse.ArgumentParser(description='parameters for reading files.')
parser.add_argument('--dataset', type=str, default='instant_video',
					choices=['instant_video', 'digital_music', 'beer'])
parser.add_argument('--data_store_path', nargs='?', default='../data/',
					help='Input data path.')
parser.add_argument('--upper_bound', type=int, default=90,
					help='upper bound for quantile')
parser.add_argument('--lower_bound', type=int, default=00,
					help='lower bound for quantile')
parser.add_argument('--latent_factor_num', type=int, default=100,
					help='latent factor number')
args=parser.parse_args()

data_store_path = args.data_store_path
dataset = args.dataset
UPPER_BOUND = args.upper_bound
LOWER_BOUND = args.lower_bound
latent_factor_num =args.latent_factor_num

if dataset == 'instant_video':
	ds = 'Instant Video'
if dataset == 'digital_music':
	ds = 'Digital Music'
if dataset == 'beer':
	ds = 'BeerAdvocate'

# mainstreamness_path = os.path.join(data_store_path, dataset, 'mainstreamness.csv')
# hist_path = os.path.join(data_store_path, 'hist_bins.txt')
train_path = os.path.join(data_store_path, dataset)
# print(os.path.abspath(mainstreamness_path))

def read_main(file_name):
    mainstreamness = []
    for i in range(4):
        mainstreamness.append(np.array(pd.read_csv(file_name).iloc[:, i+2]).tolist())
    return mainstreamness 

def read_rmse(data_path, weight_u, weight_i):
	df = pd.read_csv(os.path.join(data_path, 'user_'+str(weight_u)+'_item_'+str(weight_i), 'valid_test_rmse_bounded.csv'), header=0, index_col=None)
	columns = ['user', '1', '777', '1234', '1476', '1573', '1771', '1842', '1992', '2003', '2020']
	df = df[columns]
	return df

def read_valid(data_path, weight_u, weight_i):
	columns = ['user', '1', '777', '1234', '1476', '1573', '1771', '1842', '1992', '2003', '2020']
	df = pd.read_csv(os.path.join(data_path, 'user_'+str(weight_u)+'_item_'+str(weight_i), 'valid_rmse_bounded.csv'), header=0, index_col=None, usecols=columns)	
	df = df[columns]
	return df

def read_test(data_path, weight_u, weight_i):
	columns = ['user', '1', '777', '1234', '1476', '1573', '1771', '1842', '1992', '2003', '2020']
	df = pd.read_csv(os.path.join(data_path, 'user_'+str(weight_u)+'_item_'+str(weight_i), 'test_rmse_bounded.csv'), header=0, index_col=None, usecols=columns)
	
	df = df[columns]
	return df

def read_cos(data_path, weight_u, weight_i):
	df = pd.read_csv(os.path.join(data_path, 'user_'+str(weight_u)+'_item_'+str(weight_i), 'full_cos_sim.csv'), header=0, index_col=None)
	columns = ['user', '1', '777', '1234', '1476', '1573', '1771', '1842', '1992', '2003', '2020']
	df = df[columns]
	return df

def read_train(data_path, seed):
	df = pd.read_csv(os.path.join(data_path, 'NAECF_'+str(seed), 'train_user_item_rating.csv'), header=0, index_col=None)
	return df


# mainstreamness = read_main(mainstreamness_path)

# mainstreamness_0 = mainstreamness[0]
# mainstreamness_1 = mainstreamness[1]
# mainstreamness_2 = mainstreamness[2]
# mainstreamness_3 = mainstreamness[3]

seeds = [1, 777, 1234, 1476, 1573, 1771, 1842, 1992, 2003, 2020]
weights_u = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
weights_i = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]


data_path = os.path.join(data_store_path, dataset, 'Results', 'latent_factor_'+str(latent_factor_num), 'mode_4')
mf_path = os.path.join(data_store_path, dataset, 'Results', 'latent_factor_'+str(latent_factor_num), 'mode_5')
re_mf_path = os.path.join(data_store_path, dataset, 'Results', 'latent_factor_'+str(latent_factor_num), 'mode_5_')
df_mf = pd.read_csv(os.path.join(mf_path, 'test_rmse_bounded.csv'), header=0, index_col=None)
df_re_mf = pd.read_csv(os.path.join(re_mf_path, 'test_rmse_bounded.csv'), header=0, index_col=None)
columns = ['user', '1', '777', '1234', '1476', '1573', '1771', '1842', '1992', '2003', '2020']
df_mf = df_mf[columns]
df_re_mf = df_re_mf[columns]

# data for weight=0; re-run
if dataset == 'beer':
	data_re_path = os.path.join(data_store_path, dataset, 'Results', 'latent_factor_'+str(latent_factor_num), 'mode_4')
	df_re = pd.read_csv(os.path.join(data_re_path, 'user_'+str(0.0)+'_item_'+str(0.0)+'_', 'valid_test_rmse_bounded.csv'), header=0, index_col=None)
	df_re_val = pd.read_csv(os.path.join(data_re_path, 'user_'+str(0.0)+'_item_'+str(0.0)+'_', 'valid_rmse_bounded.csv'), header=0, index_col=None)
	df_re_test = pd.read_csv(os.path.join(data_re_path, 'user_'+str(0.0)+'_item_'+str(0.0)+'_', 'test_rmse_bounded.csv'), header=0, index_col=None)
	columns = ['user', '1', '777', '1234', '1476', '1573', '1771', '1842', '1992', '2003', '2020']
	df_re = df_re[columns]
	df_re_val = df_re_val[columns]
	df_re_test = df_re_test[columns]

else:
	data_re_path = os.path.join('../../NAECF_cos_copy', 'data', dataset, 'Results', 'latent_factor_'+str(latent_factor_num), 'mode_4')
	df_re = pd.read_csv(os.path.join(data_re_path, 'user_'+str(0.0)+'_item_'+str(0.0), 'valid_test_rmse_bounded.csv'), header=0, index_col=None)
	df_re_val = pd.read_csv(os.path.join(data_re_path, 'user_'+str(0.0)+'_item_'+str(0.0), 'valid_rmse_bounded.csv'), header=0, index_col=None)
	df_re_test = pd.read_csv(os.path.join(data_re_path, 'user_'+str(0.0)+'_item_'+str(0.0), 'test_rmse_bounded.csv'), header=0, index_col=None)
	columns = ['user', '1', '777', '1234', '1476', '1573', '1771', '1842', '1992', '2003', '2020']
	df_re = df_re[columns]
	df_re_val = df_re_val[columns]
	df_re_test = df_re_test[columns]



dfs = []
dfs_valid = []
dfs_test = []
dfs_cos = []
for weight_u in weights_u:
	for weight_i in weights_i:
		if weight_u == 0 or weight_i == 0 or weight_u == weight_i:
			df = read_rmse(data_path, weight_u, weight_i)
			df_valid = read_valid(data_path, weight_u, weight_i)
			df_test = read_test(data_path, weight_u, weight_i)
			df_cos = read_cos(data_path, weight_u, weight_i)
			dfs.append(df)
			dfs_valid.append(df_valid)
			dfs_test.append(df_test)
			dfs_cos.append(df_cos)
			print(weight_u, weight_i)



data_item_0 = []
data_item_1 = []

data_item_2 = []
data_item_3 = []

data = []
for k in range(1, 11):
	user = read_train(train_path, seeds[k-1]).user_id.tolist()
	c_user = Counter(user)
	data_t = []
	data_t_ = []
	seed = seeds[k-1]

	x = dfs_test[0].iloc[:, k].tolist()
	y = df_re_mf.iloc[:, k].tolist()
	# if i==0 and j==0:
	# 	y = dfs_test[-1].iloc[:, k].tolist()
	# else:
	# 	y = dfs_test[i*4+j].iloc[:, k].tolist()
	z = df_mf.iloc[:, k].tolist()
	for m in range(len(z)):
		if z[m] >= 0 and c_user[m] >= 3:
			data_t.append(z[m])
	th_90 = np.quantile(np.array(data_t), 0.90)
	th_10 = np.quantile(np.array(data_t), 0.10)
	th_50 = np.quantile(np.array(data_t), 0.50)

	for m in range(len(x)):
		if z[m] >= 0 and c_user[m] >= 3:
			if z[m] <= th_10 and z[m] >= 0:
				Bin = 0
				gain = y[m] - x[m]
			if z[m] <= th_50 and z[m] > th_10:
				Bin = 1
				gain = y[m] - x[m]
			if z[m] < th_90 and z[m] >= th_50:
				Bin = 2
				gain = y[m] - x[m]
			if z[m] >= th_90:
				Bin = 3
				gain = y[m] - x[m]
		else:
			Bin = -1
			gain = -100



		data.append([seed, m, c_user[m], x[m], y[m], z[m], Bin, gain])

df = pd.DataFrame(data, columns=['seed', 'user', 'count', 'uRMSE_DC', 'uRMSE_MF_re', 'uRMSE_MF', 'bin', 'gain'])
print(df.head())
print(np.array(data).shape)		
df.to_csv(os.path.join('res_data', 'Fig3_'+ds+'.csv'), header=True, index=0)	







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
plt.rcParams["figure.figsize"] = (15, 10)
plt.yticks(size = 28)
plt.xticks(size = 28)


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

# mainstreamness_path = os.path.join(data_store_path, dataset, 'mainstreamness.csv')
# hist_path = os.path.join(data_store_path, 'hist_bins.txt')
# print(os.path.abspath(mainstreamness_path))

if dataset == 'instant_video':
	ds = 'Instant Video'
if dataset == 'digital_music':
	ds = 'Digital Music'
if dataset == 'beer':
	ds = 'BeerAdvocate'

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


def read_train(data_path, seed):
	df = pd.read_csv(os.path.join(data_path, 'NAECF_'+str(seed), 'train_user_item_rating.csv'), header=0, index_col=None)
	return df

def read_valid_test(data_path, seed):
	df = pd.read_csv(os.path.join(data_path, 'NAECF_'+str(seed), 'valid_test_user_item_rating.csv'), header=0, index_col=None)
	return df


# mainstreamness = read_main(mainstreamness_path)

# mainstreamness_0 = mainstreamness[0]
# mainstreamness_1 = mainstreamness[1]
# mainstreamness_2 = mainstreamness[2]
# mainstreamness_3 = mainstreamness[3]

seeds = [1, 777, 1234, 1476, 1573, 1771, 1842, 1992, 2003, 2020]
weights_u = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
weights_i = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

datasets = ['instant_video', 'digital_music', 'beer']
markers = ['o', 'x', 'v']
colors = ['r', 'g', 'b']
labels = ['Instant Video', 'Digital Music', 'BeerAdvocate']


# if dataset == 'music_instruments':
# 	group_p = [1, 777, 1234, 1476, 1573, 1771, 2003]
# 	user_size = 1429
# 	bad = [2020]
# if dataset == 'digital_music':
# 	group_p = [1, 777, 1234, 1842, 2003, 2020]
# 	user_size = 5541
# if dataset == 'instant_video':
# 	group_p = [1, 777, 1234, 1476, 1573, 1842, 1992, 2003, 2020]
# 	user_size = 5130
# if dataset == 'beer':
# 	group_p = [777, 1476, 1771, 1842, 1992, 2020]
# 	user_size = 3703

# data for weight=0; re-run
i = 0
for dataset in datasets:
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
		df_re= pd.read_csv(os.path.join(data_re_path, 'user_'+str(0.0)+'_item_'+str(0.0), 'valid_test_rmse_bounded.csv'), header=0, index_col=None)
		# df_re_test = pd.read_csv(os.path.join(data_re_path, 'user_'+str(0.0)+'_item_'+str(0.0), 'test_rmse.csv'), header=0, index_col=None)
		df_re_val = pd.read_csv(os.path.join(data_re_path, 'user_'+str(0.0)+'_item_'+str(0.0), 'valid_rmse_bounded.csv'), header=0, index_col=None)
		df_re_test = pd.read_csv(os.path.join(data_re_path, 'user_'+str(0.0)+'_item_'+str(0.0), 'test_rmse_bounded.csv'), header=0, index_col=None)
		columns = ['user', '1', '777', '1234', '1476', '1573', '1771', '1842', '1992', '2003', '2020']
		# df_re = df_re[columns]
		df_re_val = df_re_val[columns]
		df_re_test = df_re_test[columns]

	#
	# group_n = sorted(list(set(seeds) - set(group_p)))


	data_path = os.path.join(data_store_path, dataset, 'Results', 'latent_factor_'+str(latent_factor_num), 'mode_4')
	train_path = os.path.join(data_store_path, dataset)


# for seed in seeds:
# 	print(str(seed) + ':')
# 	df_train = read_train(train_path, seed)
# 	df_valid_test = read_valid_test(train_path, seed)
# 	user = df_train.user_id.tolist()
# 	user_ = df_valid_test.user_id.tolist()
# 	rating = df_valid_test.rating.tolist()

# 	c_user = Counter(user)
# 	c_user_ = Counter(user_)

# 	rating_removal = []
# 	for u in range(len(user_)):
# 		if c_user[user_[u]] >= 3 and c_user_[user_[u]] >= 3:
# 			rating_removal.append(rating[u])

# 	n, (smin, smax), sm, sv, ss, sk = describe(rating)
# 	sstr = '%-14s num = %6.4d, variance = %6.4f, skew = %6.4f, kurtosis = %6.4f'
# 	print(sstr % ('distribution:', n, sv, ss ,sk))

# 	n, (smin, smax), sm, sv, ss, sk = describe(rating_removal)
# 	sstr = '%-14s num = %6.4d, variance = %6.4f, skew = %6.4f, kurtosis = %6.4f'
# 	print(sstr % ('distribution:', n, sv, ss ,sk))

	dfs = []
	dfs_valid = []
	dfs_test = []
	for weight_u in weights_u:
		for weight_i in weights_i:
			if weight_u == 0 or weight_i == 0 or weight_u == weight_i:
				df = read_rmse(data_path, weight_u, weight_i)
				df_valid = read_valid(data_path, weight_u, weight_i)
				df_test = read_test(data_path, weight_u, weight_i)
				dfs.append(df)
				dfs_valid.append(df_valid)
				dfs_test.append(df_test)
				# print(weight_u, weight_i)

	gain_ind = []
	gain_ind_90 = []


	for k in range(1, 11):
		# j indicates the weight
		user = read_train(train_path, seeds[k-1]).user_id.tolist()
		user_ = read_valid_test(train_path, seeds[k-1]).user_id.tolist()
		c_user = Counter(user)
		c_user_ = Counter(user_)
		gains = []
		gains_90 = []
		for j in range(8):
			data_user_item = []
			data = []
			data_bin_0 = []
			data_bin_1 = []
			data_bin_2 = []
			data_bin_3 = []
			x = dfs_valid[0].iloc[:, k].tolist()
			if j == 0:
				y = df_re_val.iloc[:, k].tolist()
			else:
				y = dfs_valid[2*j+7].iloc[:, k].tolist()
			for m in range(len(x)):

				if x[m] >= 0 and c_user[m] >= 3:
					data_user_item.append([x[m], x[m]-y[m]])

			th_90 = np.quantile(np.array(data_user_item)[:, 0], 0.90)
			th_10 = np.quantile(np.array(data_user_item)[:, 0], 0.10)
			th_50 = np.quantile(np.array(data_user_item)[:, 0], 0.50)
			th_65 = np.quantile(np.array(data_user_item)[:, 0], 0.65)
			for pair in data_user_item:
				if pair[0] <= th_10 and pair[0] >= 0:
					data_bin_0.append(pair)
				if pair[0] >= th_10 and pair[0] < th_50:
					data_bin_1.append(pair)
				if pair[0] >= th_50 and pair[0] < th_90:
					data_bin_2.append(pair)
				if pair[0] >= th_90:
					data_bin_3.append(pair)
			# gam = LinearGAM(n_splines=5).fit(np.array(data_user_item)[:, 0], np.array(data_user_item)[:, 1])
			# XX = gam.generate_X_grid(term=0)
			# print(str(seeds[k-1]) + '\t' + str(weights_u[j]))

			gain = 0.1 * np.mean(np.array(data_bin_0)[:, 1]) + \
				   0.4 * np.mean(np.array(data_bin_1)[:, 1]) + \
				   0.4 * np.mean(np.array(data_bin_2)[:, 1]) + \
				   0.1 * np.mean(np.array(data_bin_3)[:, 1])
			gain_90 =0.1 * np.mean(np.array(data_bin_0)[:, 1]) + \
				   0.4 * np.mean(np.array(data_bin_1)[:, 1]) + \
				   0.4 * np.mean(np.array(data_bin_2)[:, 1])
			gains.append(gain)
			gains_90.append(gain_90)	 

			if j == 7:
				gain_ind.append(np.argmax(gains))
				gain_ind_90.append(np.argmax(gains_90))
				indices = (-np.array(gains)).argsort()[:3]
				print([weights_u[i] for i in indices])


	# print(gain_ind)
	# print(gain_ind_90)

		# # fig, axes = plt.subplots(len(seeds), len(weights_u), sharey=True, sharex=True, figsize=(20, 20))
	gain_abs = []
	gain_abs_90 = []

	for k in range(1, 11):
		# j indicates the weight
		user = read_train(train_path, seeds[k-1]).user_id.tolist()
		user_ = read_valid_test(train_path, seeds[k-1]).user_id.tolist()
		c_user = Counter(user)
		c_user_ = Counter(user_)
		gains = []
		gains_90 = []
		
		for j in range(8):
			data_user_item = []
			data = []
			data_bin_0 = []
			data_bin_1 = []
			data_bin_2 = []
			data_bin_3 = []
			x = dfs_test[0].iloc[:, k].tolist()
			if j == 0:
				y = df_re_test.iloc[:, k].tolist()
			else:
				y = dfs_test[2*j+7].iloc[:, k].tolist()
			for m in range(len(x)):

				if x[m] >= 0 and c_user[m] >= 3:
					data_user_item.append([x[m], x[m]-y[m]])

			th_90 = np.quantile(np.array(data_user_item)[:, 0], 0.90)
			th_10 = np.quantile(np.array(data_user_item)[:, 0], 0.10)
			th_50 = np.quantile(np.array(data_user_item)[:, 0], 0.50)
			th_65 = np.quantile(np.array(data_user_item)[:, 0], 0.65)
			for pair in data_user_item:
				if pair[0] <= th_10 and pair[0] >= 0:
					data_bin_0.append(pair)
				if pair[0] >= th_10 and pair[0] < th_50:
					data_bin_1.append(pair)
				if pair[0] >= th_50 and pair[0] < th_90:
					data_bin_2.append(pair)
				if pair[0] >= th_90:
					data_bin_3.append(pair)
			# gam = LinearGAM(n_splines=5).fit(np.array(data_user_item)[:, 0], np.array(data_user_item)[:, 1])
			# XX = gam.generate_X_grid(term=0)
			# print(str(seeds[k-1]) + '\t' + str(weights_u[j]))

			gain = 0.1 * np.mean(np.array(data_bin_0)[:, 1]) + \
				   0.4 * np.mean(np.array(data_bin_1)[:, 1]) + \
				   0.4 * np.mean(np.array(data_bin_2)[:, 1]) + \
				   0.1 * np.mean(np.array(data_bin_3)[:, 1])
			gain_90 = 0.1 * np.mean(np.array(data_bin_0)[:, 1]) + \
				   0.4 * np.mean(np.array(data_bin_1)[:, 1]) + \
				   0.4 * np.mean(np.array(data_bin_2)[:, 1])	 

			gains.append(gain)
			gains_90.append(gain_90)
			if j == 7:
				# print(np.argmax(gains))

				# print(np.array(gains).argsort()[-3:][::-1])
				gain_abs.append(gains[gain_ind[k-1]]-gains[0])
				gain_abs_90.append(gains_90[gain_ind[k-1]]-gains_90[0])

	
	axis = range(10)

	plt.plot(axis, np.sort(gain_abs), c=colors[i], lw=3, label=labels[i], marker=markers[i], markersize=12)
	i += 1
	# plt.plot(axis, np.sort(gain_abs_90), c='b', lw=2, label='Top 90%', marker='o')
diag = np.linspace(0, 9.2)
zeros = np.linspace(0, 0)
plt.plot(diag, zeros, c='black', lw=1, ls='--')
ax=plt.gca();#获得坐标轴的句柄
ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(2);####设置上部坐标轴的粗细
plt.xlabel('Splits (sorted by performance gain)', fontsize=28)
plt.ylabel('Performance Gain',fontsize=28)
plt.legend(loc='upper left', fontsize=28)
# plt.title(ds, fontsize=16, fontweight='bold')
plt.tight_layout()
plt.yticks()
# plt.xticks(axis, seeds)
plt.xticks([])
plt.savefig(os.path.join('figures', 'q2_camera_ready' + '.png'),dpi=300)
			
			
print(gain_abs)
print(gain_abs_90)
print(np.mean(gain_abs), np.mean(gain_abs_90))
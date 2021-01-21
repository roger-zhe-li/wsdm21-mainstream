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
plt.rc('xtick',labelsize=28,)
plt.rc('ytick',labelsize=28, )


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

def read_cos(data_path, weight_u, weight_i):
	df = pd.read_csv(os.path.join(data_path, 'user_'+str(weight_u)+'_item_'+str(weight_i), 'full_cos_sim.csv'), header=0, index_col=None)
	columns = ['user', '1', '777', '1234', '1476', '1573', '1771', '1842', '1992', '2003', '2020']
	df = df[columns]
	return df

def read_cos_item(data_path, weight_u, weight_i):
	df = pd.read_csv(os.path.join(data_path, 'user_'+str(weight_u)+'_item_'+str(weight_i), 'full_cos_sim_item.csv'), header=0, index_col=None)
	columns = ['item', '1', '777', '1234', '1476', '1573', '1771', '1842', '1992', '2003', '2020']
	df = df[columns]
	return df

def read_point_error(data_path, weight_u, weight_i, seed):
	df = pd.read_csv(os.path.join(data_path, 'user_'+str(weight_u)+'_item_'+str(weight_i), 'seed_'+str(seed), 'valid_test_error_bounded.csv'), header=0, index_col=None)
	return df


def read_train(data_path, seed):
	df = pd.read_csv(os.path.join(data_path, 'NAECF_'+str(seed), 'train_user_item_rating.csv'), header=0, index_col=None)
	return df

def read_valid_test(data_path, seed):
	df = pd.read_csv(os.path.join(data_path, 'NAECF_'+str(seed), 'valid_test_user_item_rating.csv'), header=0, index_col=None)
	return df

seeds = [1, 777, 1234, 1476, 1573, 1771, 1842, 1992, 2003, 2020]
weights_u = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
weights_i = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]


data_path = os.path.join(data_store_path, dataset, 'Results', 'latent_factor_'+str(latent_factor_num), 'mode_4')
train_path = os.path.join(data_store_path, dataset)


dfs = []
dfs_valid = []
dfs_test = []
dfs_cos = []
dfs_cos_item = []
for weight_u in weights_u:
	for weight_i in weights_i:
		if weight_u == 0 or weight_i == 0 or weight_u == weight_i:
			df = read_rmse(data_path, weight_u, weight_i)
			# df_valid = read_valid(data_path, weight_u, weight_i)
			df_test = read_test(data_path, weight_u, weight_i)
			df_cos = read_cos(data_path, weight_u, weight_i)
			df_cos_item = read_cos_item(data_path, weight_u, weight_i)
			dfs.append(df)
			# dfs_valid.append(df_valid)
			dfs_test.append(df_test)
			dfs_cos.append(df_cos)
			dfs_cos_item.append(df_cos_item)
			print(weight_u, weight_i)


data = []
data_90 = []

# fig, axes = plt.subplots(1, len(weights_u)-1, sharey=True, sharex=True, figsize=(70, 12))
# for k in range(1, 11):
# 	if seeds[k-1] == 1771:
# 		for j in range(1, 8):
# 			data = []
# 			data_90 = []

# 			x = dfs_test[2*j+7].iloc[:, k].tolist()
# 			y = dfs_cos[2*j+7].iloc[:, k].tolist()

# 			user = read_train(train_path, seeds[k-1]).user_id.tolist()
# 			user_ = read_valid_test(train_path, seeds[k-1]).user_id.tolist()
# 			c_user = Counter(user)
# 			c_user_ = Counter(user_)

# 			for m in range(len(x)):
# 				if x[m] >= 0 and y[m] > -1 and c_user[m] >= 3:
# 					data.append([x[m], y[m]])
					
# 			th_90 = np.quantile(np.array(data)[:, 0], 0.90)
# 			for pair in data:
# 				if pair[0] <= th_90 and pair[0] >= 0:
# 					data_90.append(pair)

# 			gam = LinearGAM(n_splines=4).fit(np.array(data)[:, 0], np.array(data)[: ,1])
# 			XX = gam.generate_X_grid(term=0)
# 			gam_90 = LinearGAM(n_splines=4).fit(np.array(data_90)[:, 0], np.array(data_90)[: ,1])
# 			XX_90 = gam_90.generate_X_grid(term=0)
# 			axes[j-1].plot(XX, gam.predict(XX), color='b', lw=3)
# 			axes[j-1].plot(XX_90, gam_90.predict(XX_90), color='r',lw=3)
# 			axes[j-1].scatter(np.array(data)[:, 0], np.array(data)[:, 1], color='gray', s=8)

# 			axes[j-1].spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
# 			axes[j-1].spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
# 			axes[j-1].spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
# 			axes[j-1].spines['top'].set_linewidth(2);####
# 			axes[j-1].tick_params(axis='both', direction='out', labelsize=28)
# 			# axes[j-1].set_ylim(0.4, 0.5)


# for ax, weight in zip(axes, weights_u[1:]):
# 	ax.set_title('w='+str(weight), fontsize=32, fontweight='heavy')
# fig = axes[0].get_figure()  # getting the figure
# ax0 = fig.add_subplot(111, frame_on=False)   # creating a single axes
# plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
# plt.xlabel('RMSE', fontsize=32)
# plt.ylabel('User Cosine Similarity',fontsize=32, labelpad=40)
# # fig.suptitle(ds, fontsize=36, fontweight='bold', y=1.05)
# plt.title(ds, fontsize=36, fontweight='bold', y=1.08)
# plt.tight_layout()
# fig.savefig('q2_'+dataset+'_2.png',dpi=300)

fig, axes = plt.subplots(1, 7, sharey=True, sharex=True, figsize=(70, 12))
for j in range(1, 8):
	data = []
	data_90 = []	
	data_bin_0 = []
	data_bin_1 = []
	data_bin_2 = []
	data_bin_3 = []	
	for k in range(1, 11):
		data_t = []
		x = dfs_test[2*j+7].iloc[:, k].tolist()
		y = dfs_cos[2*j+7].iloc[:, k].tolist()

		user = read_train(train_path, seeds[k-1]).user_id.tolist()
		user_ = read_valid_test(train_path, seeds[k-1]).user_id.tolist()
		c_user = Counter(user)
		c_user_ = Counter(user_)

		for m in range(len(x)):
			if x[m] >= 0 and y[m] > -1 and c_user[m] >= 3:
				data.append([np.log1p(x[m]), y[m]])
				data_t.append([np.log1p(x[m]), y[m]])
				
		th_90 = np.quantile(np.array(data_t)[:, 0], 0.90)
		th_10 = np.quantile(np.array(data_t)[:, 0], 0.10)
		th_50 = np.quantile(np.array(data_t)[:, 0], 0.50)
		# print(th_90, th_10, th_50)

		for pair in data_t:
			# if pair[0] <= th_90 and pair[0] >= 0:
			# 	data_90.append(pair)
			if pair[0] >= 0 and pair[0] < th_10:
				data_bin_0.append(pair)
			if pair[0] >= th_10 and pair[0] < th_50:
				data_bin_1.append(pair)
			if pair[0] >= th_50 and pair[0] < th_90:
				data_bin_2.append(pair)
			if pair[0] >= th_90:
				data_bin_3.append(pair)

	gam = GammaGAM(n_splines=4).fit(np.array(data)[:, 0], np.array(data)[: ,1])
	XX = gam.generate_X_grid(term=0)
	# gam_90 = LinearGAM(n_splines=4).fit(np.array(data_90)[:, 0], np.array(data_90)[: ,1])
	# XX_90 = gam_90.generate_X_grid(term=0)
	axes[j-1].plot(XX, gam.predict(XX), color='black', ls='--', lw=3, label='All data')
	# axes[j-1].plot(XX_90, gam_90.predict(XX_90), color='r',lw=3)
	# axes[k-1, j-1].scatter(np.array(data)[:, 0], np.array(data)[:, 1], color='gray', s=8)
	gam_0 = GammaGAM(n_splines=4).fit(np.array(data_bin_0)[:, 0], np.array(data_bin_0)[: ,1])
	XX_0 = gam_0.generate_X_grid(term=0)
	axes[j-1].plot(XX_0, gam_0.predict(XX_0), lw=3, label='bin_0')

	gam_1 = GammaGAM(n_splines=4).fit(np.array(data_bin_1)[:, 0], np.array(data_bin_1)[: ,1])
	XX_1 = gam_1.generate_X_grid(term=0)
	axes[j-1].plot(XX_1, gam_1.predict(XX_1), lw=3, label='bin_1')

	gam_2 = GammaGAM(n_splines=4).fit(np.array(data_bin_2)[:, 0], np.array(data_bin_2)[: ,1])
	XX_2 = gam_2.generate_X_grid(term=0)
	axes[j-1].plot(XX_2, gam_2.predict(XX_2), lw=3, label='bin_2')

	gam_3 = GammaGAM(n_splines=4).fit(np.array(data_bin_3)[:, 0], np.array(data_bin_3)[: ,1])
	XX_3 = gam_3.generate_X_grid(term=0)
	axes[j-1].plot(XX_3, gam_3.predict(XX_3), lw=3, label='bin_3')


	axes[j-1].scatter(np.array(data)[:, 0], np.array(data)[:, 1], color='gray', s=1)

	axes[j-1].spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
	axes[j-1].spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
	axes[j-1].spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
	axes[j-1].spines['top'].set_linewidth(2);####
	axes[j-1].tick_params(axis='both', direction='out', labelsize=28)
	axes[j-1].set_rasterized(True)
	if dataset == 'instant_video':
		axes[j-1].set_ylim(0.2, 0.4)
	elif dataset == 'digital_music':
		axes[j-1].set_ylim(0.15, 0.3)
	elif dataset == 'beer':
		axes[j-1].set_ylim(0.42, 0.46)

	if j == 1:
		axes[j-1].legend(loc='upper_right', fontsize=24)


	# for ax, seed in zip(axes[:, -1], seeds):
	# 	ax.yaxis.set_label_position("right")
	# 	ax.set_ylabel(str(seed), fontsize=7)
	    
for ax, weight in zip(axes, weights_u[1:]):
	ax.set_title('w='+str(weight), fontsize=32, fontweight='heavy')
	
fig = axes[0].get_figure()  # getting the figure
ax0 = fig.add_subplot(111, frame_on=False)   # creating a single axes
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel('uRMSE', fontsize=32)
plt.ylabel('User Cosine Similarity',fontsize=32, labelpad=40)
# fig.suptitle(ds, fontsize=36, fontweight='bold', y=1.05)
plt.title(ds, fontsize=36, fontweight='bold', y=1.08)
plt.tight_layout()
fig.savefig(os.path.join('figures', 'q3_'+dataset+'.png',dpi=300))







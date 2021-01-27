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
from matplotlib import cm 
from matplotlib.collections import LineCollection
from matplotlib.ticker import ScalarFormatter

formatter = ScalarFormatter()
formatter.set_scientific(False)


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

datasets = ['instant_video', 'digital_music', 'beer']
ds = ['Instant Video', 'Digital Music', 'BeerAdvocate']

# mainstreamness_path = os.path.join(data_store_path, dataset, 'mainstreamness.csv')
# hist_path = os.path.join(data_store_path, 'hist_bins.txt')
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

map_vir = cm.get_cmap(name='Set1')
color = [map_vir(i) for i in np.linspace(0, 1, 4)]
# print(color)



fig, axes = plt.subplots(1, 3, sharex=True, figsize=(50, 20), dpi=80)
for i in range(len(datasets)):
	dataset = datasets[i]
	data_path = os.path.join(data_store_path, dataset, 'Results', 'latent_factor_'+str(latent_factor_num), 'mode_4')
	train_path = os.path.join(data_store_path, dataset)
	dfs = []
	dfs_valid = []
	dfs_test = []
	dfs_cos = []
	dfs_cos_item = []
	for weight_u in weights_u:
		for weight_i in weights_i:
			if weight_u == weight_i:
				df = read_rmse(data_path, weight_u, weight_i)
				# df_valid = read_valid(data_path, weight_u, weight_i)
				df_test = read_test(data_path, weight_u, weight_i)
				df_cos = read_cos(data_path, weight_u, weight_i)
				df_cos_items = read_cos_item(data_path, weight_u, weight_i)
				dfs.append(df)
				# dfs_valid.append(df_valid)
				dfs_test.append(df_test)
				dfs_cos.append(df_cos)
				dfs_cos_item.append(df_cos_items)
				print(weight_u, weight_i)

	data = []

	data_bin_0 = []
	data_bin_1 = []
	data_bin_2 = []
	data_bin_3 = []	


	for j in range(1, 8):		
		for k in range(1, 11):
			data_t = []
			x = dfs_test[j].iloc[:, k].tolist()
			y = dfs_cos[j].iloc[:, k].tolist()
			y = [((1 - value) / 2) ** 2 for value in y] 

			user = read_train(train_path, seeds[k-1]).user_id.tolist()
			user_ = read_valid_test(train_path, seeds[k-1]).user_id.tolist()
			c_user = Counter(user)
			c_user_ = Counter(user_)

			for m in range(len(x)):
				if x[m] >= 0 and y[m] > -1 and c_user[m] >= 3:
					# data.append([np.log1p(x[m]), y[m]])
					data_t.append([1+x[m], y[m]])
					
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

	axes[i].set_rasterization_zorder(1)
	axes[i].scatter(np.array(data_bin_0)[:, 0], np.array(data_bin_0)[:, 1], s=4, alpha=.03, zorder=0, color=color[0])
	axes[i].scatter(np.array(data_bin_1)[:, 0], np.array(data_bin_1)[:, 1], s=4, alpha=.03, zorder=0, color=color[1])
	axes[i].scatter(np.array(data_bin_2)[:, 0], np.array(data_bin_2)[:, 1], s=4, alpha=.03, zorder=0, color=color[2])
	axes[i].scatter(np.array(data_bin_3)[:, 0], np.array(data_bin_3)[:, 1], s=4, alpha=.03, zorder=0, color=color[3])

	axes[i].errorbar(x=np.mean(np.array(data_bin_0)[:, 0]), y=np.mean(np.array(data_bin_0)[:, 1]), xerr=np.std(np.array(data_bin_0)[:, 0], ddof=1), yerr=np.std(np.array(data_bin_0)[:, 1], ddof=1), fmt='o:', elinewidth=6, capsize=12, label='Bin1', color=color[0])
	axes[i].errorbar(x=np.mean(np.array(data_bin_1)[:, 0]), y=np.mean(np.array(data_bin_1)[:, 1]), xerr=np.std(np.array(data_bin_1)[:, 0], ddof=1), yerr=np.std(np.array(data_bin_1)[:, 1], ddof=1), fmt='o:', elinewidth=6, capsize=12, label='Bin2', color=color[1])
	axes[i].errorbar(x=np.mean(np.array(data_bin_2)[:, 0]), y=np.mean(np.array(data_bin_2)[:, 1]), xerr=np.std(np.array(data_bin_2)[:, 0], ddof=1), yerr=np.std(np.array(data_bin_2)[:, 1], ddof=1), fmt='o:', elinewidth=6, capsize=12, label='Bin3', color=color[2])
	axes[i].errorbar(x=np.mean(np.array(data_bin_3)[:, 0]), y=np.mean(np.array(data_bin_3)[:, 1]), xerr=np.std(np.array(data_bin_3)[:, 0], ddof=1), yerr=np.std(np.array(data_bin_3)[:, 1], ddof=1), fmt='o:', elinewidth=6, capsize=12, label='Bin4', color=color[3])
	
	
	if i == 0:
		axes[i].set_ylim(0.08, 0.18)
	elif i == 1:
		axes[i].set_ylim(0.10, 0.20)
	elif i == 2:
		axes[i].set_ylim(0.07, 0.09)
	axes[i].set_xscale('log', basex=np.e)
	axes[i].xaxis.set_major_formatter(formatter)
	axes[i].set_xticks(range(1, 6))
	axes[i].set_xticklabels(range(5))


	axes[i].spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
	axes[i].spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
	axes[i].spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
	axes[i].spines['top'].set_linewidth(2);####
	axes[i].tick_params(axis='both', direction='out', labelsize=36)
	axes[i].grid()
	axes[i].set_title(ds[i], fontsize=56, fontweight='bold', y=1.02)


# axes[1].legend(loc=2, bbox_to_anchor=(-0.2, -0.1), borderaxespad=0, fontsize=48,  ncol=4)


	# for ax, seed in zip(axes[:, -1], seeds):
	# 	ax.yaxis.set_label_position("right")
	# 	ax.set_ylabel(str(seed), fontsize=7)
	    
# for ax, weight in zip(axes, weights_u[1:]):
# 	ax.set_title('w='+str(weight), fontsize=32, fontweight='heavy')
fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.2)
# axes.flatten()[-2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4)
axes.flatten()[-2].legend(loc='lower center', bbox_to_anchor=(0.5, -0.17), borderaxespad=0, fontsize=36,  ncol=4)
	
fig = axes[0].get_figure()  # getting the figure
ax0 = fig.add_subplot(111, frame_on=False)   # creating a single axes
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel('uRMSE', fontsize=48)
plt.ylabel('User Reconstruction Loss',fontsize=48, labelpad=48)
# fig.suptitle(ds, fontsize=36, fontweight='bold', y=1.05)
# plt.title(ds, fontsize=36, fontweight='bold', y=1.02)
plt.tight_layout()
# fig.savefig(os.path.join('figures', 'bin_camera_ready.png'), rasterized=True, dpi=300, bbox_inches='tight')
fig.savefig(os.path.join('figures', 'bin_scatter.png'), rasterized=True)
# fig.savefig(os.path.join('figures', 'q3_camera_ready.png'), rasterized=True, dpi=300)

# print(np.mean(np.array(data_bin_0)[:, 0]), np.mean(np.array(data_bin_0)[:, 1]))
# print(np.mean(np.array(data_bin_1)[:, 0]), np.mean(np.array(data_bin_1)[:, 1]))
# print(np.mean(np.array(data_bin_2)[:, 0]), np.mean(np.array(data_bin_2)[:, 1]))
# print(np.mean(np.array(data_bin_3)[:, 0]), np.mean(np.array(data_bin_3)[:, 1]))



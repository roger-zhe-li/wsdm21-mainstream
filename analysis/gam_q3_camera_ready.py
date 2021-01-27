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
# matplotlib.rcParams['agg.path.chunksize'] = 10000


warnings.filterwarnings("ignore")
matplotlib.use('Agg')
np.set_printoptions(suppress=True)
plt.rcParams["figure.figsize"] = (3, 2)
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

# if dataset == 'instant_video':
# 	ds = 'Instant Video'
# if dataset == 'digital_music':
# 	ds = 'Digital Music'
# if dataset == 'beer':
# 	ds = 'BeerAdvocate'

datasets = ['instant_video', 'digital_music', 'beer']
ds = ['Instant Video', 'Digital Music', 'BeerAdvocate']

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

map_vir = cm.get_cmap(name='jet')
color = [map_vir(i) for i in np.linspace(0, 1, len(weights_u))]
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


	for j in range(1, 8):
		data = []
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
					data.append([1+x[m], y[m]])
					# data_t.append([np.log1p(x[m]), y[m]])
			# print(th_90, th_10, th_50)

		gam = LinearGAM(n_splines=4).fit(np.array(data)[:, 0], np.array(data)[: ,1])
		XX = gam.generate_X_grid(term=0)
		# gam_90 = LinearGAM(n_splines=4).fit(np.array(data_90)[:, 0], np.array(data_90)[: ,1])
		# XX_90 = gam_90.generate_X_grid(term=0)
		if i == 0:
			axes[i].plot(XX, gam.predict(XX), color=color[j], lw=3, label=str(weights_u[j]))
		else:
			axes[i].plot(XX, gam.predict(XX), color=color[j], lw=3)
		# axes.set_ylim(0.41, 0.47)
		# if i == 0:
		# 	axes[i].set_ylim(0.15, 0.40)
		# elif i == 1:
		# 	axes[i].set_ylim(0.10, 0.35)
		# elif i == 2:
		# 	axes[i].set_ylim(0.41, 0.47)
		axes[i].set_xscale('log', basex=np.e)
		axes[i].xaxis.set_major_formatter(formatter)
		axes[i].set_xticks(range(1, 6))
		axes[i].set_xticklabels(range(5))
		axes[i].set_title(ds[i], fontsize=56, fontweight='bold', y=1.02)
		# axes[j-1].plot(XX_90, gam_90.predict(XX_90), color='r',lw=3)
		# axes[k-1, j-1].scatter(np.array(data)[:, 0], np.array(data)[:, 1], color='gray', s=8)
		
		# axes[i].set_rasterization_zorder(1)
		axes[i].scatter(np.array(data)[:, 0], np.array(data)[:, 1], alpha=.1, s=4, color=color[j], zorder=0)
		


		axes[i].spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
		axes[i].spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
		axes[i].spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
		axes[i].spines['top'].set_linewidth(2);####
		axes[i].tick_params(axis='both', direction='out', labelsize=36)
		axes[i].grid()

		if i == 0:
			axes[i].set_ylim(0.07, 0.2)
		elif i == 1:
			axes[i].set_ylim(0.07, 0.2)
		elif i == 2:
			axes[i].set_ylim(0.06, 0.1)


# if dataset == 'instant_video':
# 	axes[j-1].set_ylim(0.2, 0.4)
# elif dataset == 'digital_music':
# 	axes[j-1].set_ylim(0.15, 0.3)
# elif dataset == 'beer':
# 	axes[j-1].set_ylim(0.42, 0.46)

	# if dataset == 'beer':
	# 	axes[j-1].set_ylim(0.06, 0.1)

# axes[-1].legend(loc=2, bbox_to_anchor=(1.05, 0.7), borderaxespad=0, fontsize=48, title='Weight', ncol=1, title_fontsize=48)
fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.2)
# axes.flatten()[-2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4)
fig.legend(loc='lower center', bbox_to_anchor=(0.55, -0.01), borderaxespad=0, fontsize=36,  ncol=8, title_fontsize=36)

	# for ax, seed in zip(axes[:, -1], seeds):
	# 	ax.yaxis.set_label_position("right")
	# 	ax.set_ylabel(str(seed), fontsize=7)
	    
# for ax, weight in zip(axes, weights_u[1:]):
# 	ax.set_title('w='+str(weight), fontsize=32, fontweight='heavy')
	
fig = axes[0].get_figure()  # getting the figure
ax0 = fig.add_subplot(111, frame_on=False)   # creating a single axes
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel('uRMSE', fontsize=48)
plt.ylabel('User Reconstruction Loss',fontsize=48, labelpad=48)
# fig.suptitle(ds, fontsize=36, fontweight='bold', y=1.05)
# plt.title(ds, fontsize=36, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join('figures', 'q3_scatter_new.png'), rasterized=True, dpi=80)
# fig.savefig(os.path.join('figures', 'q3_camera_ready.png'), rasterized=True, dpi=300)







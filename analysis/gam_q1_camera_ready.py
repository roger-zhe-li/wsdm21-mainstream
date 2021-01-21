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

mainstreamness_path = os.path.join(data_store_path, dataset, 'mainstreamness.csv')
hist_path = os.path.join(data_store_path, 'hist_bins.txt')
train_path = os.path.join(data_store_path, dataset)
# print(os.path.abspath(mainstreamness_path))

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

def read_train(data_path, seed):
	df = pd.read_csv(os.path.join(data_path, 'NAECF_'+str(seed), 'train_user_item_rating.csv'), header=0, index_col=None)
	return df


mainstreamness = read_main(mainstreamness_path)

mainstreamness_0 = mainstreamness[0]
mainstreamness_1 = mainstreamness[1]
mainstreamness_2 = mainstreamness[2]
mainstreamness_3 = mainstreamness[3]

seeds = [1, 777, 1234, 1476, 1573, 1771, 1842, 1992, 2003, 2020]
weights_u = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
weights_i = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]


map_vir = cm.get_cmap(name='Reds')
color = [map_vir(i) for i in np.linspace(0, 1, len(weights_u))]
print(color)

opt_weights = [[2, 5, 0.1, 10, 1, 5, 0.1, 0.1, 0, 0.5],
				[0, 0, 5, 5, 2, 0.1, 2, 0.1, 0.1, 0.5],
				[0.1, 0.2, 0, 0.5, 0, 0.2, 0.5, 0.1, 0.1, 0.2]]




fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(45, 15))
for i in range(len(datasets)):
	dataset = datasets[i]

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
			if weight_u == weight_i:
				df = read_rmse(data_path, weight_u, weight_i)
				df_valid = read_valid(data_path, weight_u, weight_i)
				df_test = read_test(data_path, weight_u, weight_i)
				df_cos = read_cos(data_path, weight_u, weight_i)
				dfs.append(df)
				dfs_valid.append(df_valid)
				dfs_test.append(df_test)
				dfs_cos.append(df_cos)
				print(weight_u, weight_i)



	data_mf = []

	data_dc = []
	data_best = []
	for k in range(1, 11):
		best = weights_u.index(opt_weights[i][k-1])
	# 	if opt_weights[k] 
		user = read_train(train_path, seeds[k-1]).user_id.tolist()
		c_user = Counter(user)

		x = dfs_test[0].iloc[:, k].tolist()
		y = df_re_mf.iloc[:, k].tolist()
		b = dfs_test[best].iloc[:, k].tolist()

	# 	# if i==0 and j==0:
	# 	# 	y = dfs_test[-1].iloc[:, k].tolist()
	# 	# else:
	# 	# 	y = dfs_test[i*4+j].iloc[:, k].tolist()
		z = df_mf.iloc[:, k].tolist()
		for m in range(len(x)):
			if x[m] >= 0 and c_user[m] >= 3:
				data_dc.append([1+z[m], z[m]-x[m]])
				data_best.append([1+z[m], z[m]-b[m]])
				# if j == 0:
				data_mf.append([1+z[m], z[m]-y[m]])	

	gam = LinearGAM(n_splines=4).fit(np.array(data_dc)[:, 0], np.array(data_dc)[:, 1])
	XX = gam.generate_X_grid(term=0)

	# # # gam_90 = LinearGAM(n_splines=4).fit(np.array(data_item_1)[:, 0], np.array(data_item_1)[:, 1])
	# # # XX_90 = gam_90.generate_X_grid(term=0)
	u_lim = max(max(np.array(data_dc)[:, 0]), max(np.array(data_dc)[:, 1]))
	diag = np.linspace(1, 1+u_lim)
	zeros = np.linspace(0, 0)

	# axes[i].scatter(np.array(data_dc)[:, 0], np.array(data_dc)[:, 1], s=4, alpha=.03, color='r')
	# axes[i].scatter(np.array(data_best)[:, 0], np.array(data_best)[:, 1], s=4, alpha=.03, color='black')
	# axes[i].scatter(np.array(data_mf)[:, 0], np.array(data_mf)[:, 1], s=4, alpha=.03, color='b')

	axes[i].plot(diag, zeros, color='black', ls='--', lw=3)

	if i == 0:
		axes[i].plot(XX, gam.predict(XX), label='DeepCoNN', color='r', lw=4)
	else:
		axes[i].plot(XX, gam.predict(XX), color='r', lw=4)
	# plt.plot(XX_90, gam_90.predict(XX_90), color='r', ls='--', lw=3, label='Top 90% uRMSE, X=DeepCoNN')

	gam_best = LinearGAM(n_splines=4).fit(np.array(data_best)[:, 0], np.array(data_best)[:, 1])
	XX_best = gam_best.generate_X_grid(term=0)
	# if i == 0:
	# 	axes[i].plot(XX_best, gam_best.predict(XX), label='NAECF_best', color='black', lw=2)
	# else:
	# 	axes[i].plot(XX_best, gam_best.predict(XX), color='black', lw=2)

	gam_mf = LinearGAM(n_splines=4).fit(np.array(data_mf)[:, 0], np.array(data_mf)[:, 1])
	XX = gam_mf.generate_X_grid(term=0)
	# # # gam_mf_90 = LinearGAM(n_splines=4).fit(np.array(data_item_3)[:, 0], np.array(data_item_3)[:, 1])
	# # # XX_90 = gam_mf_90.generate_X_grid(term=0)
	if i == 0:
		axes[i].plot(XX, gam_mf.predict(XX), color='b', label='MF retraining', lw=4)
	else:
		axes[i].plot(XX, gam_mf.predict(XX), color='b', lw=4)
	# # # plt.plot(XX_90, gam_mf_90.predict(XX_90), color='b', ls='--', lw=3, label='Top 90% uRMSE, X=MF_re')
	axes[i].set_xscale('log', basex=np.e)
	axes[i].xaxis.set_major_formatter(formatter)
	axes[i].set_xticks(range(1, 6))
	axes[i].set_xticklabels(range(5))
	axes[i].set_xlim(1, 5)

	axes[i].spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
	axes[i].spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
	axes[i].spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
	axes[i].spines['top'].set_linewidth(2);####
	axes[i].tick_params(axis='both', direction='out', labelsize=36)
	axes[i].grid()
	axes[i].set_title(ds[i], fontsize=56, fontweight='bold', y=1.02)

fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.2)
# axes.flatten()[-2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4)
# axes.flatten()[-2].legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), borderaxespad=0, fontsize=36,  ncol=4)
fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.01), borderaxespad=0, fontsize=36,  ncol=3, title_fontsize=36)

	
fig = axes[0].get_figure()  # getting the figure
ax0 = fig.add_subplot(111, frame_on=False)   # creating a single axes
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel('uRMSE_MF', fontsize=48, labelpad=40)
plt.ylabel('uRMSE gain over MF',fontsize=48, labelpad=60)
# fig.suptitle(ds, fontsize=36, fontweight='bold', y=1.05)
# plt.title(ds, fontsize=36, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join('figures', 'q1_camera_ready.png'), rasterized=True, dpi=300, bbox_inches='tight')
# fig.savefig(os.path.join('figures', 'q1_scatter.png'), rasterized=True, dpi=80, bbox_inches='tight')



# ax=plt.gca();#获得坐标轴的句柄
# ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
# ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
# ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
# ax.spines['top'].set_linewidth(2);####设置上部坐标轴的粗细

# # gam = LinearGAM(n_splines=4).fit(np.array(data_item_1)[:, 0], np.array(data_item_1)[:, 1])
# # XX = gam.generate_X_grid(term=0)
# # axes[i, j].plot(XX, gam.predict(XX), color='blue', lw=.5, label='NAECF')
# # axes[i, j].plot(XX, gam.confidence_intervals(XX), color='black', ls='--', lw=1)
# # plt.scatter(np.array(data_item_0)[:, 0], np.array(data_item_0)[:, 1], color='red', s=0.5, alpha=.1)
# # plt.scatter(np.array(data_item_2)[:, 0], np.array(data_item_2)[:, 1], color='blue', s=0.5, alpha=.1)
# plt.xlabel('uRMSE.MF', fontsize=12)
# plt.ylabel('uRMSE.MF - uRMSE.X',fontsize=12)
# plt.yticks([])
# plt.legend(loc='lower left', fontsize=12)
# plt.title(ds, fontsize=16, fontweight='bold')
# # plt.yticks(rotation=45)
# plt.tight_layout()
# plt.savefig(os.path.join('figures', 'q1_camera_ready'+'.png'),dpi=300)








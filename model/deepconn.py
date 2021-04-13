# -*- coding: utf-8 -*-
import copy
import json
import math
import os
import random

import msgpack
from tqdm import tqdm
import lmdb
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn
from torch.utils.data import DataLoader, Dataset
import logging
from util import data_split_pandas
from util.npl_util import WordVector
import torch.nn.utils.rnn as rnn_utils
import torchsnooper
import scipy.stats

np.set_printoptions(suppress=True)
logger = logging.getLogger('RATE.DeepCoNN.train_test')


class FactorizationMachine(nn.Module):

	def __init__(self, factor_size: int, fm_k: int):
		super(FactorizationMachine, self).__init__()
		self._linear = nn.Linear(factor_size, 1)
		self._v = torch.nn.Parameter(torch.randn((factor_size, fm_k)))
		self._drop = nn.Dropout(0.2)

	def forward(self, x):
		# linear regression
		w = self._linear(x).squeeze()

		# cross feature
		inter1 = torch.matmul(x, self._v)
		inter2 = torch.matmul(x**2, self._v**2)
		inter = (inter1**2 - inter2) * 0.5
		# inter = self._drop(inter)
		inter = torch.sum(inter, dim=1)

		return w + inter


class Encoder(nn.Module):

	def __init__(self, conv_kernel_num: int, conv_length: int, word_vec_dim: int, 
				review_length: int, latent_factor_num: int):
		super(Encoder, self).__init__()
		self._conv = nn.Conv2d(in_channels=1,
							   out_channels=conv_kernel_num,
							   kernel_size=(conv_length, word_vec_dim))
		self._maxpool = nn.MaxPool2d(kernel_size=(review_length-conv_length+1, 1), return_indices=True)
		self._flatten = Flatten()
		self._linear = nn.Linear(conv_kernel_num, latent_factor_num)
		self._nonlinear = nn.ReLU()


	def forward(self, x):
		conv_0 = self._conv(x)
		relu_0 = self._nonlinear(conv_0)
		maxpool, indices = self._maxpool(relu_0)
		flatten = self._flatten(maxpool)
		fc = self._linear(flatten)
		emb = self._nonlinear(fc)

		return emb, indices


class Decoder(nn.Module):
	def __init__(self, conv_kernel_num: int, conv_length: int, word_vec_dim: int,
				review_length: int, latent_factor_num: int):
		super(Decoder, self).__init__()
		self._linear = nn.Linear(latent_factor_num, conv_kernel_num)
		self._nonlinear = nn.ReLU()
		self._maxunpool = nn.MaxUnpool2d(kernel_size=(review_length-conv_length+1, 1))
		self._unflatten = UnFlatten()
		self._deconv = nn.ConvTranspose2d(in_channels=conv_kernel_num,
										  out_channels=1,
										  kernel_size=(conv_length, word_vec_dim))

	def forward(self, x, indices):
		fc = self._linear(x)
		fc_ = self._unflatten(fc)
		maxunpool = self._maxunpool(fc_, indices)
		relu = self._nonlinear(maxunpool)
		emb_restore = self._deconv(relu)

		return emb_restore


class Flatten(nn.Module):
	"""
	squeeze layer for Sequential structure
	"""
	def forward(self, x):
		return x.squeeze()


class UnFlatten(nn.Module):
	def forward(self, x):
		return x.view(x.size()[0], x.size()[1], 1, 1)


def collate_fn(data):
	data.sort(key=lambda x: len(x), reverse=True)
	data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
	return data


class DeepCoNNDataLoader:

	def __init__(self, data_path: str, batch_size, lmdb_path, zero_index: int,
				 review_length: int, device: torch.device, shuffle=False):
		self._data = pd.read_csv(data_path)\
						 .reset_index(drop=True) \
						 .loc[:, ['user_id', 'item_id', 'rating']].to_numpy()

		self._lmdb = lmdb.open(os.path.join(lmdb_path), readonly=True)
		self._zero_index = zero_index
		self._review_length = review_length
		self._device = device
		self._shuffle = shuffle
		self._batch_size = batch_size
		self._index = 0
		self._index_list = list(range(self._data.shape[0]))
		if shuffle:
			random.shuffle(self._index_list)

	def __len__(self):
		return math.ceil(len(self._index_list) // self._batch_size)
		# return len(self._data)

	def __iter__(self):
		return self

	def __next__(self):
		if self._index < len(self._index_list):
			# data = self._data.loc[self._index: self._index+self._batch_size-1]
			idx_ls = self._index_list[self._index: self._index+self._batch_size]
			self._index += self._batch_size

			data = self._data[idx_ls, :]
			users = data[:, 0].tolist()
			items = data[:, 1].tolist()
			rating = data[:, 2].astype(np.float32)

			users_ = ['user-{:d}'.format(int(u)).encode() for u in users]
			items_ = ['item-{:d}'.format(int(u)).encode() for u in items]

			with self._lmdb.begin() as txn:
				user_review_tokens = [txn.get(u) for u in users_]
				item_review_tokens = [txn.get(i) for i in items_]
#            user_review_tokens = [msgpack.unpackb(u, use_list=False, raw=False) for u in user_review_tokens]
#            item_review_tokens = [msgpack.unpackb(i, use_list=False, raw=False) for i in item_review_tokens]
				user_review_tokens = map(lambda x: msgpack.unpackb(x), user_review_tokens)
				item_review_tokens = map(lambda x: msgpack.unpackb(x), item_review_tokens)

				user_review_tokens = [np.asarray(t[:self._review_length], dtype=np.longlong) for t in user_review_tokens]
				item_review_tokens = [np.asarray(t[:self._review_length], dtype=np.longlong) for t in item_review_tokens]
				
				user_review_tokens = [torch.from_numpy(x).to(self._device) for x in user_review_tokens]
				item_review_tokens = [torch.from_numpy(x).to(self._device) for x in item_review_tokens]
				
				user_review_tokens = self._pad(user_review_tokens)
				item_review_tokens = self._pad(item_review_tokens)
				
				rating = torch.from_numpy(rating).to(self._device)
				
				return users, items, user_review_tokens, item_review_tokens, rating
		else:
			self._index = 0
			raise StopIteration

	def _pad(self, tensor_list):
		out_tensor = tensor_list[0].data.new_full(
			(len(tensor_list), self._review_length), self._zero_index)
		# for i, tensor in enumerate(out_tensor):
		for i, tensor in enumerate(tensor_list):
			length = tensor.size(0)
			out_tensor[i, :length, ...] = tensor

		return out_tensor

	def close_lmdb(self):
		self._lmdb.close()


# class DeepCoNNDataSet(Dataset):
#
#     def __init__(self, data: pd.DataFrame, folder, zero_index: int,
#                  review_length: int, device: torch.device):
#         # self._data = data.reset_index(drop=True)
#         self._data = data.to_numpy()
#
#         self._lmdb = lmdb.open(os.path.join(folder, 'lmdb'), readonly=True)
#
#         self._zero_index = zero_index
#         self._review_length = review_length
#         self._device = device
#
#     def __len__(self):
#         return len(self._data)
#
#     # noinspection PyArgumentList
#     def __getitem__(self, x):
#         """
#         :param x:
#         :return: review token ids of users and items with fixed length
#         """
#         # uir = self._data.loc[x].to_dict()
#         uir = self._data[x]
#         # user = 'user-{:d}'.format(uir['user_id'])
#         # item = 'item-{:d}'.format(uir['item_id'])
#         # rating = uir['rating']
#         user = 'user-{:d}'.format(uir[2])
#         item = 'item-{:d}'.format(uir[4])
#         rating = uir[5]
#
#         with self._lmdb.begin() as txn:
#             user_review_tokens = msgpack.unpackb(txn.get(str(user).encode()))
#             item_review_tokens = msgpack.unpackb(txn.get(str(item).encode()))
#
#         user_tokens = user_review_tokens[:self._review_length]
#         item_tokens = item_review_tokens[:self._review_length]
#
#         user_tokens = torch.Tensor(user_tokens)
#         user_tokens = self._pad(user_tokens).long().to(self._device)
#         item_tokens = torch.Tensor(item_tokens)
#         item_tokens = self._pad(item_tokens).long().to(self._device)
#
#         rating = torch.FloatTensor([rating]).to(self._device)
#
#         return user_tokens, item_tokens, rating
#
#     def close_lmdb(self):
#         self._lmdb.close()
#
#     def _pad(self, vector: torch.Tensor):
#         if vector.size(0) < self._review_length:
#             return \
#                 torch.cat((vector,
#                            vector.new_full([self._review_length-vector.size(0)],
#                                            self._zero_index)))
#         elif vector.size(0) > self._review_length:
#             return vector[:self._review_length]
#         else:
#             return vector


class DataPreFetcher:

	def __init__(self, loader):
		self._loader = iter(loader)
		self._rating = None
		self._item_review = None
		self._user_review = None
		self.pre_load()

	def pre_load(self):
		try:
			self._user_review, self._item_review, self._rating \
				= next(self._loader)
		except StopIteration:
			self._rating = None
			self._item_review = None
			self._user_review = None
			return

	def next(self):
		# data = self._next_data
		user_review = self._user_review
		item_review = self._item_review
		rating = self._rating
		self.pre_load()
		return user_review, item_review, rating


class DeepCoNN(nn.Module):

	def __init__(self, review_length, word_vec_dim, fm_k, 
				 conv_length,
				 conv_kernel_num, latent_factor_num):
		"""
		:param review_length: 评论单词数
		:param word_vec_dim: 词向量维度
		:param conv_length: 卷积核的长度
		:param conv_kernel_num: 卷积核数量
		:param latent_factor_num: 全连接输出的特征维度
		"""
		super(DeepCoNN, self).__init__()

		# input_shape: (batch_size, 1, review_length, word_vec_dim)
		self.__encoder = Encoder(conv_kernel_num, conv_length, word_vec_dim, 
				review_length, latent_factor_num)
		self.__decoder = Decoder(conv_kernel_num, conv_length, word_vec_dim,
				review_length, latent_factor_num)

		# input: (batch_size, 2*latent_factor_num)
		self.__factor_machine = FactorizationMachine(latent_factor_num * 2,
													 fm_k)

	def forward(self, user_review, item_review):
		user_latent, user_indices = self.__encoder(user_review)
		item_latent, item_indices = self.__encoder(item_review)
		user_emb_restore = self.__decoder(user_latent, user_indices)
		item_emb_restore = self.__decoder(item_latent, item_indices)

		# concatenate
		concat_latent = torch.cat((user_latent, item_latent), dim=1)
		# print(concat_latent.is_cuda)
		prediction = self.__factor_machine(concat_latent)

		return prediction, user_emb_restore, item_emb_restore


# def get_review_average_length(df: pd.DataFrame, review_column: str):
#     df['sentences_length'] = df[review_column].apply(lambda x: len(x))
#     return df['sentences_length'].mean()


# noinspection PyUnreachableCode,PyArgumentList
class DeepCoNNTrainTest:

	# noinspection PyUnresolvedReferences
	def __init__(self, epoch, batch_size, dir_path, device, 
				review_length, word_vector_dim, conv_length, conv_kernel_num, fm_k, latent_factor_num,
				 learning_rate, save_folder, random_state, coef, coef_u, coef_i, mode, dataset):
		"""
		训练，测试DeepCoNN
		"""
		self._epoch = epoch
		self._batch_size = batch_size
		self._review_length = review_length
		self._dir_path = os.path.join(dir_path, 'NAECF_'+str(random_state))
		self._device = torch.device(device)
		self._save_dir = os.path.join(dir_path, 'NAECF_'+str(random_state), save_folder, str(coef))        
		self._res_dir = os.path.join(dir_path, 'Results', 'latent_factor_'+str(latent_factor_num), str(coef), 'seed_'+str(random_state))
		self._conv_len = conv_length
		self._coef = coef
		self._random_state = random_state
		self._coef_u = coef_u
		self._coef_i = coef_i
		self._mode = mode
		self._dataset = dataset



		logger.info('epoch:{:<8d} batch size:{:d}'.format(epoch, batch_size))

		self.lmdb_dir = os.path.join('/home/nfs/zli6', self._dataset, 'NAECF_'+str(self._random_state))

		# if mode == 0:
		#     self._save_dir = os.path.join(dir_path, 'NAECF_'+str(random_state), save_folder, 'mode_0', str(coef))
		#     self._res_dir = os.path.join(dir_path, 'Results', 'latent_factor_'+str(latent_factor_num), 'mode_0', str(coef), 'seed_'+str(random_state))
		#     self._lmdb_path = os.path.join(self.lmdb_dir, 'mode_0', 'lmdb_'+str(self._coef))
		# if mode == 1:
		#     self._save_dir = os.path.join(dir_path, 'NAECF_'+str(random_state), save_folder, 'mode_1', str(coef))
		#     self._res_dir = os.path.join(dir_path, 'Results', 'latent_factor_'+str(latent_factor_num), 'mode_1', str(coef), 'seed_'+str(random_state))
		#     self._lmdb_path = os.path.join(self.lmdb_dir, 'mode_1', 'lmdb_'+str(self._coef))
		# if mode == 2:
		#     self._save_dir = os.path.join(dir_path, 'NAECF_'+str(random_state), save_folder, 'mode_2', 'user_'+str(coef_u)+'_item_'+str(coef_i))
		#     self._res_dir = os.path.join(dir_path, 'Results', 'latent_factor_'+str(latent_factor_num), 'mode_2', 'user_'+str(coef_u)+'_item_'+str(coef_i), 'seed_'+str(random_state))
		#     self._lmdb_path = os.path.join(self.lmdb_dir, 'mode_2', 'lmdb_user_'+str(self._coef_u)+'_item_'+str(self._coef_i))
		# if mode == 3: 
		#     self._save_dir = os.path.join(dir_path, 'NAECF_'+str(random_state), save_folder, 'mode_3', 'user_'+str(coef_u)+'_item_'+str(coef_i))
		#     self._res_dir = os.path.join(dir_path, 'Results', 'latent_factor_'+str(latent_factor_num), 'mode_3', 'user_'+str(coef_u)+'_item_'+str(coef_i), 'seed_'+str(random_state))
		#     self._lmdb_path = os.path.join(self.lmdb_dir, 'mode_3', 'lmdb_user_'+str(self._coef_u)+'_item_'+str(self._coef_i))
		if mode == 4: 
			self._save_dir = os.path.join(dir_path, 'NAECF_'+str(random_state), save_folder, 'mode_4', 'user_'+str(coef_u)+'_item_'+str(coef_i))
			self._res_dir = os.path.join(dir_path, 'Results', 'latent_factor_'+str(latent_factor_num), 'mode_4', 'user_'+str(coef_u)+'_item_'+str(coef_i), 'seed_'+str(random_state))
			# rerun
			# self._save_dir = os.path.join(dir_path, 'NAECF_'+str(random_state), save_folder, 'mode_4', 'user_'+str(coef_u)+'_item_'+str(coef_i)+'_')
			# self._res_dir = os.path.join(dir_path, 'Results', 'latent_factor_'+str(latent_factor_num), 'mode_4', 'user_'+str(coef_u)+'_item_'+str(coef_i)+'_', 'seed_'+str(random_state))
			self._lmdb_path = os.path.join(self.lmdb_dir, 'mode_4', 'lmdb_user_'+str(coef_u)+'_item_'+str(coef_i))
			print(self._lmdb_path)


		# print(self._lmdb_path)
		if not os.path.exists(self._save_dir):
			os.makedirs(self._save_dir)
		if not os.path.exists(self._res_dir):
			os.makedirs(self._res_dir)
		if not os.path.exists(self._lmdb_path):
			os.makedirs(self._lmdb_path)

		# read data
		self._train_data = pd.read_csv(
			os.path.join(self._dir_path,  'train_user_item_rating.csv'))
		self._test_data = pd.read_csv(
			os.path.join(self._dir_path, 'test_user_item_rating.csv'))

		with open(os.path.join(self._dir_path, 'dataset_meta_info.json'),
				  'r') as f:
			dataset_meta_info = json.load(f)
		self._user_size = dataset_meta_info['user_size']
		self._item_size = dataset_meta_info['item_size']
		self._dataset_size = dataset_meta_info['dataset_size']
		# print(self._review_length)
		self._review_length = self._review_length
		# print(self._review_length)

		# initial DeepCoNN model
		self._model = DeepCoNN(review_length=self._review_length,
							   word_vec_dim=word_vector_dim,
							   fm_k=fm_k,
							   conv_length=self._conv_len,
							   conv_kernel_num=conv_kernel_num,
							   latent_factor_num=latent_factor_num,
							   ).to(self._device)

		self._optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self._model.parameters()),
										   lr=learning_rate)
		# self._loss_func = torch.nn.MSELoss()
		self._loss_func = torch.nn.MSELoss(reduction='none')
		self._cos_loss_fn = torch.nn.CosineEmbeddingLoss(reduction='none')


		# user and item count
		user_ids = self._train_data.user_id.tolist()
		user_count = []
		for i in range(self._user_size):
			user_count.append(user_ids.count(i))

		item_ids = self._train_data.item_id.tolist()
		item_count = []
		for j in range(self._item_size):
			item_count.append(item_ids.count(j))


		embedding_user = torch.FloatTensor(user_count).unsqueeze(dim=1)
		self._count_emb_user \
			= nn.Embedding.from_pretrained(embedding_user).requires_grad_(False).to(self._device)
		# print(self._count_emb_user(torch.LongTensor([100]).to(self._device)))
		# print(user_count[100])
		embedding_item = torch.FloatTensor(item_count).unsqueeze(dim=1)
		self._count_emb_item \
			= nn.Embedding.from_pretrained(embedding_item).requires_grad_(False).to(self._device)

		# load pretrained embedding
		logger.info('Initialize word embedding model for pytorch')
		embedding = torch.FloatTensor(WordVector().vectors)
		zero_tensor = torch.zeros(size=embedding[:1].size())
		self._zero_index = embedding.size()[0]
		embedding = torch.cat((embedding, zero_tensor), dim=0)
		self._embedding \
			= nn.Embedding.from_pretrained(embedding).requires_grad_(False).to(self._device)

		logger.info('Model initialized, start training...')

		# dataloader
		logger.info('Initialize dataloader.')


	# @torchsnooper.snoop()
	def train(self):
		# data = pd.read_csv('{}/train_user_item_rating.csv'
		#                    .format(self._dir_path))

		# train_data, valid_data = data_split_pandas(data, 0.9, 0.1)

		train_data_path = os.path.join(self._dir_path,
									   'train_user_item_rating.csv')
		valid_data_path = os.path.join(self._dir_path,
									   'valid_user_item_rating.csv')
		test_data_path = os.path.join(self._dir_path,
									   'test_user_item_rating.csv')
		train_valid_data_path = os.path.join(self._dir_path,
									   'train_valid_user_item_rating.csv')
		valid_test_data_path = os.path.join(self._dir_path,
									   'valid_test_user_item_rating.csv')
		full_data_path = os.path.join(self._dir_path,
									   'full_rating.csv')
		# if self._mode == 0: 
		#     self._lmdb_path = os.path.join(self._dir_path, 'mode_0', 'lmdb_'+str(self._coef))
		# if self._mode == 1:
		#     self._lmdb_path = os.path.join(self._dir_path, 'mode_1', 'lmdb_'+str(self._coef))
		# if self._mode == 2:
		#     self._lmdb_path = os.path.join(self._dir_path, 'mode_2', 'lmdb_user_'+str(self._coef_u)+'_item_'+str(self._coef_i))

		lmdb_path = self._lmdb_path
		train_data_loader = DeepCoNNDataLoader(data_path=train_data_path,
											   batch_size=self._batch_size,
											   lmdb_path=lmdb_path,
											   zero_index=self._zero_index,
											   review_length=self._review_length,
											   device=self._device,
											   shuffle=True)

		valid_data_loader = DeepCoNNDataLoader(data_path=valid_data_path,
											   batch_size=self._batch_size,
											   lmdb_path=lmdb_path,
											   zero_index=self._zero_index,
											   review_length=self._review_length,
											   device=self._device)

		logger.info('Start training.')
		best_valid_loss = float('inf')
		# best_model_state_dict = None
		best_valid_epoch = 0

		for e in tqdm(range(self._epoch)):
			# fetcher = DataPreFetcher(self._data_loader)
			# user_tokens, item_tokens, rating = fetcher.next()
			train_loss = None
			train_error = []
			train_loss_value = []
			for users, items, user_tokens, item_tokens, rating \
					in train_data_loader:
				user_review_vec = self._embedding(user_tokens).unsqueeze(dim=1)
				item_review_vec = self._embedding(item_tokens).unsqueeze(dim=1)

				pred, user_emb_restore, item_emb_restore = self._model(user_review_vec, item_review_vec)
				y = torch.Tensor([1]).to(self._device)
				target = torch.Tensor([0]).to(self._device)
				user_freq = torch.bincount(torch.LongTensor(users), minlength=self._user_size).requires_grad_(False)
				item_freq = torch.bincount(torch.LongTensor(items), minlength=self._item_size).requires_grad_(False)
				user_count = user_freq[users].to(self._device)
				item_count = item_freq[items].to(self._device)

				if self._mode == 4:
					if e < 50:
						# train_loss = self._loss_func(pred, rating.flatten())
						train_loss = 1.0 / 16 * self._loss_func(pred, rating.flatten())
					else:
						train_loss = 1.0 / 16 * self._loss_func(pred, rating.flatten()) +\
									1.0 / user_count * self._coef_u * self._loss_func(target, self._cos_loss_fn(user_review_vec.view(len(pred), -1), user_emb_restore.view(len(pred), -1), y)) +\
									1.0 / item_count * self._coef_i * self._loss_func(target, self._cos_loss_fn(item_review_vec.view(len(pred), -1), item_emb_restore.view(len(pred), -1), y))  

				# train_loss = self._loss_func(pred, rating.flatten()) + \
				#              1.0 / user_count * self._coef * weight_u * self._loss_func(user_review_vec, user_emb_restore) + \
				#              1.0 / item_count * self._coef * weight_i * self._loss_func(item_review_vec, item_emb_restore) 
				# print(train_loss.sum())   

				self._optimizer.zero_grad()
				train_loss.sum().backward()
				# print(train_loss.sum())
				self._optimizer.step()
				with torch.no_grad():
					batch_error = pred - rating.flatten()
					train_error.append(batch_error.cpu().numpy())
					train_loss_value.append(train_loss.cpu().numpy())
			train_error = np.concatenate(train_error, axis=None)**2
			train_error = train_error.mean().item()
			train_loss_value = np.concatenate(train_loss_value, axis=None)
			print(torch.cuda.memory_summary(device=None, abbreviated=True))

			# print(np.mean(train_loss_value))
			error = []
			for _, _, valid_user_tokens, valid_item_tokens, valid_rating \
					in valid_data_loader:
				user_review_vec = \
					self._embedding(valid_user_tokens).unsqueeze(dim=1)
				item_review_vec = \
					self._embedding(valid_item_tokens).unsqueeze(dim=1)

				with torch.no_grad():
					batch_pred, _, _ = self._model(user_review_vec,
											 item_review_vec)

					batch_error = batch_pred - valid_rating
					error.append(batch_error.cpu().numpy())

			error = np.concatenate(error, axis=None)**2
			error = error.mean().item()

			# if best_valid_loss > error:
			#     best_model_state_dict = copy.deepcopy(self._model.state_dict())
			#     best_valid_loss = error
			#     best_valid_epoch = e
			if e == self._epoch - 1:
				best_model_state_dict = copy.deepcopy(self._model.state_dict())
				valid_loss = error
				best_valid_epoch = e

				torch.save(best_model_state_dict,
						   os.path.join(self._save_dir,
										'NAECF.tar'))
				# print(os.path.join(self._save_dir, 'NAECF.tar'))
			logger.info(
				'epoch: {}, train loss: {:.5f}, valid loss: {:.5f}'
				.format(e, np.mean(train_loss_value), error))
			

		with open(os.path.join(self._save_dir, 'training.json'), 'w') as f:
			json.dump({'seed': self._random_state,
						'epoch': best_valid_epoch,
					   'valid_loss': np.sqrt(best_valid_loss)},
					  f)

		train_data_loader.close_lmdb()
		valid_data_loader.close_lmdb()


	def rmse_analysis(self, mode='test'):
		if mode == 'test':
			data_path = os.path.join(self._dir_path, 'test_user_item_rating.csv')
		elif mode == 'valid':
			data_path = os.path.join(self._dir_path, 'valid_user_item_rating.csv')
		elif mode == 'valid_test':
			data_path = os.path.join(self._dir_path, 'valid_test_user_item_rating.csv')
		elif mode == 'train_valid':
			data_path = os.path.join(self._dir_path, 'train_valid_user_item_rating.csv')
		elif mode == 'train':
			data_path = os.path.join(self._dir_path, 'train_user_item_rating.csv')
		elif mode == 'full':
			data_path = os.path.join(self._dir_path, 'full_rating.csv')

		lmdb_path = self._lmdb_path
		data_loader = DeepCoNNDataLoader(data_path=data_path,
										 batch_size=self._batch_size,
										 lmdb_path=lmdb_path,
										 zero_index=self._zero_index,
										 review_length=self._review_length,
										 device=self._device)
		# print(os.path.abspath(os.path.join(self._save_dir, 'NAECF.tar')))
		self._model.load_state_dict(torch.load(os.path.join(self._save_dir,
															'NAECF.tar')))

		error = []
		pred = []
		uid = []
		iid = []
		rmse = []
		cos_sim = []
		cos_sim_item = []
		cos = []
		cos_item = []

		for users, items, ur, ir, r in data_loader:
			user_review_vec = \
				self._embedding(ur).unsqueeze(dim=1)
			item_review_vec = \
				self._embedding(ir).unsqueeze(dim=1)
			# y = torch.Tensor([1]).to(self._device)

			with torch.no_grad():
				batch_pred, user_emb_restore, item_emb_restore = self._model(user_review_vec,
										 item_review_vec)
				# clamp with boundings
				batch_pred = torch.clamp(batch_pred, 1, 5)

				batch_error = batch_pred - r
				# print(user_review_vec.view(len(batch_pred), -1).shape, user_emb_restore.shape)
				user_cos = torch.nn.CosineSimilarity()(user_review_vec.view(len(batch_pred), -1), user_emb_restore.view(len(batch_pred), -1))
				item_cos = torch.nn.CosineSimilarity()(item_review_vec.view(len(batch_pred), -1), item_emb_restore.view(len(batch_pred), -1))
			error.append(batch_error.cpu().numpy())
			pred.append(batch_pred.cpu().numpy())
			uid.append(users)
			iid.append(items)
			cos_sim.append(user_cos.cpu().numpy())
			cos_sim_item.append(item_cos.cpu().numpy())
		cos_sim = np.concatenate(cos_sim, axis=None)
		cos_sim_item = np.concatenate(cos_sim_item, axis=None)
		# print(len(cos_sim))
		# print(lecos_df = 

		error_ = np.concatenate(error, axis=None) ** 2
		error = error_.mean().item()
		pred = np.concatenate(pred, axis=None)
		uid = np.concatenate(uid, axis=None)
		iid = np.concatenate(iid, axis=None)
		pred_df = pd.DataFrame(list(zip(uid, iid, pred)))
		pred_df.to_csv(os.path.join(self._res_dir, mode+'_pred_bounded.csv'), index=None, header=['user', 'item', 'prediction'])

		error_df = pd.DataFrame(list(zip(uid, iid, error_)))
		error_df.to_csv(os.path.join(self._res_dir, mode+'_error_bounded.csv'), index=None, header=['user', 'item', 'error'])
		# print(max(cos_sim))
		if mode == 'full':
			# cos_all = list(zip(uid, cos_sim))
			# for i in range(self._user_size):
			# 	if i not in uid:
			# 		cos.append([i, -1.0])
			# 	else:
			# 		similarity = np.mean([x[1] for x in cos_all if x[0] == i])
			# 		cos.append([i, similarity])
			# cos = np.array(cos)
			
			# cos_path = os.path.join(os.path.dirname(self._res_dir), mode+'_cos_sim.csv')
			# if not os.path.exists(os.path.dirname(cos_path)):
			# 	os.makedirs(os.path.join(os.path.dirname(cos_path)))
			# if os.path.exists(cos_path):
			# 	cos_df = pd.read_csv(cos_path, header=0, index_col=None, engine='python', error_bad_lines=False)
			# else:
			# 	cos_df = pd.DataFrame()

			# cos_df['user'] = pd.Series(cos[:, 0].astype(int))
			# cos_df[str(self._random_state)] = pd.Series(cos[:, 1])
			# cos_df.to_csv(cos_path, header=True, index=0)


			##############################################
			cos_item_all = list(zip(iid, cos_sim_item))
			for i in range(self._item_size):
				if i not in iid:
					cos_item.append([i, -1.0])
				else:
					similarity = np.mean([x[1] for x in cos_item_all if x[0] == i])
					cos_item.append([int(i), similarity])
			cos_item = np.array(cos_item)
			
			cos_path_item = os.path.join(os.path.dirname(self._res_dir), mode+'_cos_sim_item.csv')
			if not os.path.exists(os.path.dirname(cos_path_item)):
				os.makedirs(os.path.join(os.path.dirname(cos_path_item)))
			if os.path.exists(cos_path_item):
				cos_item_df = pd.read_csv(cos_path_item, header=0, index_col=None, engine='python', error_bad_lines=False)
			else:
				cos_item_df = pd.DataFrame()

			cos_item_df['item'] = pd.Series(cos_item[:, 0].astype(int))
			cos_item_df[str(self._random_state)] = pd.Series(cos_item[:, 1])
			cos_item_df.to_csv(cos_path_item, header=True, index=0)
			####################



		# rmse_all = list(zip(uid, iid, error_))
		# for i in range(self._user_size):
		# 	if i not in uid:
		# 		rmse.append([i, -1.0])
		# 	else:
		# 		x = np.array([list(x) for x in rmse_all if x[0]==i])
		# 		rmse.append([i, np.sqrt(x[:, 2].sum()/len(x[:, 2]))])
		# rmse = np.array(rmse)

		# # rmse_df = pd.DataFrame(rmse)
		# rmse_path = os.path.join(os.path.dirname(self._res_dir), mode+'_rmse_bounded.csv')
		# if not os.path.exists(os.path.dirname(rmse_path)):
		# 	os.makedirs(os.path.join(os.path.dirname(rmse_path)))
		# if os.path.exists(rmse_path):
		# 	rmse_df = pd.read_csv(rmse_path, header=0, index_col=None, engine='python', error_bad_lines=False)
		# else:
		# 	rmse_df = pd.DataFrame()

		# # if 'user' not in rmse_df.columns:
		# rmse_df['user'] = pd.Series(rmse[:, 0].astype(int))
		# rmse_df[str(self._random_state)] = pd.Series(rmse[:, 1])

		# # print(os.path.abspath(rmse_path))
		
		# # rmse_df.columns=col
		# rmse_df.to_csv(rmse_path, header=True, index=0)
		
		# if mode == 'test':
		# 	logger.info('Test MSE: {:.5f}'.format(error))
		# 	with open(os.path.join(self._save_dir, 'test_result.json'), 'w') as f:
		# 		json.dump({'rmse': np.sqrt(error),
		# 				   'seed': self._random_state}, 
		# 				f)

		data_loader.close_lmdb()


	def test(self, mode):
		self.rmse_analysis(mode)


	def valid(self, mode):
		self.rmse_analysis(mode)


	def valid_test(self, mode):
		self.rmse_analysis(mode)


	def train_valid(self, mode):
		self.rmse_analysis(mode)


	def train_(self, mode):
		self.rmse_analysis(mode)


	def full(self, mode):
		self.rmse_analysis(mode)


	def del_db(self):
		lmdb_path = self._lmdb_path
		if os.path.exists(os.path.join(lmdb_path, 'data.mdb')):
			os.remove(os.path.join(lmdb_path, 'data.mdb'))
			os.remove(os.path.join(lmdb_path, 'lock.mdb'))




	# def test(self):
	#     test_data_path = os.path.join(self._dir_path,
	#                                   'test_user_item_rating.csv')
	#     lmdb_path = self._lmdb_path
	#     data_loader = DeepCoNNDataLoader(data_path=test_data_path,
	#                                      batch_size=self._batch_size,
	#                                      lmdb_path=lmdb_path,
	#                                      zero_index=self._zero_index,
	#                                      review_length=self._review_length,
	#                                      device=self._device)
	#     self._model.load_state_dict(torch.load(os.path.join(self._save_dir,
	#                                                         'NAECF.tar')))

	#     error = []
	#     pred = []
	#     uid = []
	#     iid = []
	#     rmse = []
	#     for users, items, ur, ir, r in data_loader:
	#         user_review_vec = \
	#             self._embedding(ur).unsqueeze(dim=1)
	#         item_review_vec = \
	#             self._embedding(ir).unsqueeze(dim=1)

	#         with torch.no_grad():
	#             batch_pred, _, _ = self._model(user_review_vec,
	#                                      item_review_vec)
	#             batch_error = batch_pred - r
	#         error.append(batch_error.cpu().numpy())
	#         pred.append(batch_pred.cpu().numpy())
	#         uid.append(users)
	#         iid.append(items)

	#     error_ = np.concatenate(error, axis=None) ** 2
	#     error = error_.mean().item()
	#     pred = np.concatenate(pred, axis=None)
	#     uid = np.concatenate(uid, axis=None)
	#     iid = np.concatenate(iid, axis=None)
	#     pred_df = pd.DataFrame(list(zip(uid, iid, pred)))
	#     pred_df.to_csv(os.path.join(self._res_dir, 'test_pred.csv'), index=None, header=['user', 'item', 'prediction'])

	#     error_df = pd.DataFrame(list(zip(uid, iid, error_)))
	#     error_df.to_csv(os.path.join(self._res_dir, 'test_error.csv'), index=None, header=['user', 'item', 'error'])

	#     rmse_all = list(zip(uid, iid, error_))
	#     for i in range(self._user_size):
	#         if i not in uid:
	#             rmse.append([i, -1.0])
	#         else:
	#             x = np.array([list(x) for x in rmse_all if x[0]==i])
	#             rmse.append([i, np.sqrt(x[:, 2].sum()/len(x[:, 2]))])
	#     rmse = np.array(rmse)

	#     # rmse_df = pd.DataFrame(rmse)
	#     rmse_path = os.path.join(os.path.dirname(self._res_dir), 'test_rmse.csv')
	#     if not os.path.exists(os.path.dirname(rmse_path)):
	#         os.makedirs(os.path.join(os.path.dirname(rmse_path)))
	#     if os.path.exists(rmse_path):
	#         rmse_df = pd.read_csv(rmse_path, header=0, index_col=None, engine='python')
	#     else:
	#         rmse_df = pd.DataFrame()

	#     if 'user' not in rmse_df.columns:
	#         rmse_df['user'] = pd.Series(rmse[:, 0].astype(int))
	#     rmse_df[str(self._random_state)] = pd.Series(rmse[:, 1])
		
	#     # rmse_df.columns=col
	#     rmse_df.to_csv(os.path.join(rmse_path), header=True, index=0)



	#     logger.info('Test MSE: {:.5f}'.format(error))
	#     with open(os.path.join(self._save_dir, 'test_result.json'), 'w') as f:
	#         json.dump({'rmse': np.sqrt(error),
	#                    'seed': self._random_state}, 
	#                 f)

	#     data_loader.close_lmdb()


	# def valid(self):
	#     test_data_path = os.path.join(self._dir_path,
	#                                   'valid_user_item_rating.csv')
	#     lmdb_path = self._lmdb_path
	#     data_loader = DeepCoNNDataLoader(data_path=test_data_path,
	#                                      batch_size=self._batch_size,
	#                                      lmdb_path=lmdb_path,
	#                                      zero_index=self._zero_index,
	#                                      review_length=self._review_length,
	#                                      device=self._device)
	#     self._model.load_state_dict(torch.load(os.path.join(self._save_dir,
	#                                                         'NAECF.tar')))

	#     error = []
	#     pred = []
	#     uid = []
	#     iid = []
	#     rmse = []
	#     for users, items, ur, ir, r in data_loader:
	#         user_review_vec = \
	#             self._embedding(ur).unsqueeze(dim=1)
	#         item_review_vec = \
	#             self._embedding(ir).unsqueeze(dim=1)

	#         with torch.no_grad():
	#             batch_pred, _, _ = self._model(user_review_vec,
	#                                      item_review_vec)
	#             batch_error = batch_pred - r
	#         error.append(batch_error.cpu().numpy())
	#         pred.append(batch_pred.cpu().numpy())
	#         uid.append(users)
	#         iid.append(items)

	#     error_ = np.concatenate(error, axis=None) ** 2
	#     error = error_.mean().item()
	#     pred = np.concatenate(pred, axis=None)
	#     uid = np.concatenate(uid, axis=None)
	#     iid = np.concatenate(iid, axis=None	#     pred_df = pd.DataFrame(list(zip(uid, iid, pred)))

	#     pred_df.to_csv(os.path.join(self._res_dir, 'valid_pred.csv'), index=None, header=['user', 'item', 'prediction'])

	#     error_df = pd.DataFrame(list(zip(uid, iid, error_)))
	#     error_df.to_csv(os.path.join(self._res_dir, 'valid_error.csv'), index=None, header=['user', 'item', 'error'])
	#     # logger.info('Test MSE: {:.5f}'.format(error))
	#     # with open(os.path.join(self._save_dir, 'test_result.json'), 'w') as f:
	#     #     json.dump({'mse': error}, f)

	#     rmse_all = list(zip(uid, iid, error_))
	#     for i in range(self._user_size):
	#         if i not in uid:
	#             rmse.append([i, -1.0])
	#         else:
	#             x = np.array([list(x) for x in rmse_all if x[0]==i])
	#             rmse.append([i, np.sqrt(x[:, 2].sum()/len(x[:, 2]))])
	#     rmse = np.array(rmse)

	#     # rmse_df = pd.DataFrame(rmse)
	#     rmse_path = os.path.join(os.path.dirname(self._res_dir), 'valid_rmse.csv')
	#     if not os.path.exists(os.path.dirname(rmse_path)):
	#         os.makedirs(os.path.join(os.path.dirname(rmse_path)))
	#     if os.path.exists(rmse_path):
	#         rmse_df = pd.read_csv(rmse_path, header=0, index_col=None, engine='python')
	#     else:
	#         rmse_df = pd.DataFrame()

	#     if 'user' not in rmse_df.columns:
	#         rmse_df['user'] = pd.Series(rmse[:, 0].astype(int))
	#     rmse_df[str(self._random_state)] = pd.Series(rmse[:, 1])
		
	#     # rmse_df.columns=col
	#     rmse_df.to_csv(os.path.join(rmse_path), header=True, index=0)


	#     data_loader.close_lmdb()


	# def valid_test(self):
	#     test_data_path = os.path.join(self._dir_path,
	#                                   'valid_test_user_item_rating.csv')
	#     lmdb_path = self._lmdb_path
	#     data_loader = DeepCoNNDataLoader(data_path=test_data_path,
	#                                      batch_size=self._batch_size,
	#                                      lmdb_path=lmdb_path,
	#                                      zero_index=self._zero_index,
	#                                      review_length=self._review_length,
	#                                      device=self._device)
	#     self._model.load_state_dict(torch.load(os.path.join(self._save_dir,
	#                                                         'NAECF.tar')))

	#     error = []
	#     pred = []
	#     uid = []
	#     iid = []
	#     rmse = []
	#     for users, items, ur, ir, r in data_loader:
	#         user_review_vec = \
	#             self._embedding(ur).unsqueeze(dim=1)
	#         item_review_vec = \
	#             self._embedding(ir).unsqueeze(dim=1)

	#         with torch.no_grad():
	#             batch_pred, _, _ = self._model(user_review_vec,
	#                                      item_review_vec)
	#             batch_error = batch_pred - r
	#         error.append(batch_error.cpu().numpy())
	#         pred.append(batch_pred.cpu().numpy())
	#         uid.append(users)
	#         iid.append(items)

	#     error_ = np.concatenate(error, axis=None) ** 2
	#     error = error_.mean().item()
	#     pred = np.concatenate(pred, axis=None)
	#     uid = np.concatenate(uid, axis=None)
	#     iid = np.concatenate(iid, axis=None)
	#     pred_df = pd.DataFrame(list(zip(uid, iid, pred)))

	#     pred_df.to_csv(os.path.join(self._res_dir, 'valid_test_pred.csv'), index=None, header=['user', 'item', 'prediction'])

	#     error_df = pd.DataFrame(list(zip(uid, iid, error_)))
	#     error_df.to_csv(os.path.join(self._res_dir, 'valid_test_error.csv'), index=None, header=['user', 'item', 'error'])
	#     # logger.info('Test MSE: {:.5f}'.format(error))
	#     # with open(os.path.join(self._save_dir, 'test_result.json'), 'w') as f:
	#     #     json.dump({'mse': error}, f)

	#     rmse_all = list(zip(uid, iid, error_))
	#     for i in range(self._user_size):
	#         if i not in uid:
	#             rmse.append([i, -1.0])
	#         else:
	#             x = np.array([list(x) for x in rmse_all if x[0]==i])
	#             rmse.append([i, np.sqrt(x[:, 2].sum()/len(x[:, 2]))])
	#     rmse = np.array(rmse)

	#     # rmse_df = pd.DataFrame(rmse)
	#     rmse_path = os.path.join(os.path.dirname(self._res_dir), 'valid_test_rmse.csv')
	#     if not os.path.exists(os.path.dirname(rmse_path)):
	#         os.makedirs(os.path.join(os.path.dirname(rmse_path)))
	#     if os.path.exists(rmse_path):
	#         rmse_df = pd.read_csv(rmse_path, header=0, index_col=None, engine='python')
	#     else:
	#         rmse_df = pd.DataFrame()

	#     if 'user' not in rmse_df.columns:
	#         rmse_df['user'] = pd.Series(rmse[:, 0].astype(int))
	#     rmse_df[str(self._random_state)] = pd.Series(rmse[:, 1])

		
	#     # rmse_df.columns=col
	#     rmse_df.to_csv(os.path.join(rmse_path), header=True, index=0)


	#     data_loader.close_lmdb()


	# def train_valid(self):
	#     test_data_path = os.path.join(self._dir_path,
	#                                   'train_valid_user_item_rating.csv')
	#     lmdb_path = self._lmdb_path
	#     data_loader = DeepCoNNDataLoader(data_path=test_data_path,
	#                                      batch_size=self._batch_size,
	#                                      lmdb_path=lmdb_path,
	#                                      zero_index=self._zero_index,
	#                                      review_length=self._review_length,
	#                                      device=self._device)
	#     self._model.load_state_dict(torch.load(os.path.join(self._save_dir,
	#                                                         'NAECF.tar')))

	#     error = []
	#     pred = []
	#     uid = []
	#     iid = []
	#     rmse = []
	#     for users, items, ur, ir, r in data_loader:
	#         user_review_vec = \
	#             self._embedding(ur).unsqueeze(dim=1)
	#         item_review_vec = \
	#             self._embedding(ir).unsqueeze(dim=1)

	#         with torch.no_grad():
	#             batch_pred, _, _ = self._model(user_review_vec,
	#                                      item_review_vec)
	#             batch_error = batch_pred - r
	#         error.append(batch_error.cpu().numpy())
	#         pred.append(batch_pred.cpu().numpy())
	#         uid.append(users)
	#         iid.append(items)

	#     error_ = np.concatenate(error, axis=None) ** 2
	#     error = error_.mean().item()
	#     pred = np.concatenate(pred, axis=None)
	#     uid = np.concatenate(uid, axis=None)
	#     iid = np.concatenate(iid, axis=None)
	#     pred_df = pd.DataFrame(list(zip(uid, iid, pred)))

	#     pred_df.to_csv(os.path.join(self._res_dir, 'train_valid_pred.csv'), index=None, header=['user', 'item', 'prediction'])

	#     error_df = pd.DataFrame(list(zip(uid, iid, error_)))
	#     error_df.to_csv(os.path.join(self._res_dir, 'train_valid_error.csv'), index=None, header=['user', 'item', 'error'])
	#     # logger.info('Test MSE: {:.5f}'.format(error))
	#     # with open(os.path.join(self._save_dir, 'test_result.json'), 'w') as f:
	#     #     json.dump({'mse': error}, f)

	#     rmse_all = list(zip(uid, iid, error_))
	#     for i in range(self._user_size):
	#         if i not in uid:
	#             rmse.append([i, -1.0])
	#         else:
	#             x = np.array([list(x) for x in rmse_all if x[0]==i])
	#             rmse.append([i, np.sqrt(x[:, 2].sum()/len(x[:, 2]))])
	#     rmse = np.array(rmse)

	#     # rmse_df = pd.DataFrame(rmse)
	#     rmse_path = os.path.join(os.path.dirname(self._res_dir), 'train_valid_rmse.csv')
	#     if not os.path.exists(os.path.dirname(rmse_path)):
	#         os.makedirs(os.path.join(os.path.dirname(rmse_path)))
	#     if os.path.exists(rmse_path):
	#         rmse_df = pd.read_csv(rmse_path, header=0, index_col=None, engine='python')
	#     else:
	#         rmse_df = pd.DataFrame()

	#     if 'user' not in rmse_df.columns:
	#         rmse_df['user'] = pd.Series(rmse[:, 0].astype(int))
	#     rmse_df[str(self._random_state)] = pd.Series(rmse[:, 1])
		
	#     # rmse_df.columns=col
	#     rmse_df.to_csv(os.path.join(rmse_path), header=True, index=0)


	#     data_loader.close_lmdb()


	# def full(self):
	#     test_data_path = os.path.join(self._dir_path,
	#                                   'full_rating.csv')
	#     lmdb_path = self._lmdb_path
	#     data_loader = DeepCoNNDataLoader(data_path=test_data_path,
	#                                      batch_size=self._batch_size,
	#                                      lmdb_path=lmdb_path,
	#                                      zero_index=self._zero_index,
	#                                      review_length=self._review_length,
	#                                      device=self._device)
	#     self._model.load_state_dict(torch.load(os.path.join(self._save_dir,
	#                                                         'NAECF.tar')))

	#     error = []
	#     pred = []
	#     uid = []
	#     iid = []
	#     rmse = []
	#     for users, items, ur, ir, r in data_loader:
	#         user_review_vec = \
	#             self._embedding(ur).unsqueeze(dim=1)
	#         item_review_vec = \
	#             self._embedding(ir).unsqueeze(dim=1)

	#         with torch.no_grad():
	#             batch_pred, _, _ = self._model(user_review_vec,
	#                                      item_review_vec)
	#             batch_error = batch_pred - r
	#         error.append(batch_error.cpu().numpy())
	#         pred.append(batch_pred.cpu().numpy())
	#         uid.append(users)
	#         iid.append(items)

	#     error_ = np.concatenate(error, axis=None) ** 2
	#     error = error_.mean().item()
	#     pred = np.concatenate(pred, axis=None)
	#     uid = np.concatenate(uid, axis=None)
	#     iid = np.concatenate(iid, axis=None)
	#     pred_df = pd.DataFrame(list(zip(uid, iid, pred)))

	#     pred_df.to_csv(os.path.join(self._res_dir, 'full_pred.csv'), index=None, header=['user', 'item', 'prediction'])

	#     error_df = pd.DataFrame(list(zip(uid, iid, error_)))
	#     error_df.to_csv(os.path.join(self._res_dir, 'full_error.csv'), index=None, header=['user', 'item', 'error'])
	#     # logger.info('Test MSE: {:.5f}'.format(error))
	#     # with open(os.path.join(self._save_dir, 'test_result.json'), 'w') as f:
	#     #     json.dump({'mse': error}, f)

	#     rmse_all = list(zip(uid, iid, error_))
	#     for i in range(self._user_size):
	#         if i not in uid:
	#             rmse.append([i, -1.0])
	#         else:
	#             x = np.array([list(x) for x in rmse_all if x[0]==i])
	#             rmse.append([i, np.sqrt(x[:, 2].sum()/len(x[:, 2]))])
	#     rmse = np.array(rmse)

	#     # rmse_df = pd.DataFrame(rmse)
	#     rmse_path = os.path.join(os.path.dirname(self._res_dir), 'full_rmse.csv')
	#     if not os.path.exists(os.path.dirname(rmse_path)):
	#         os.makedirs(os.path.join(os.path.dirname(rmse_path)))
	#     if os.path.exists(rmse_path):
	#         rmse_df = pd.read_csv(rmse_path, header=0, index_col=None, engine='python')
	#     else:
	#         rmse_df = pd.DataFrame()

	#     if 'user' not in rmse_df.columns:
	#         rmse_df['user'] = pd.Series(rmse[:, 0].astype(int))
	#     rmse_df[str(self._random_state)] = pd.Series(rmse[:, 1])
		
	#     # rmse_df.columns=col
	#     rmse_df.to_csv(os.path.join(rmse_path), header=True, index=0)


	#     data_loader.close_lmdb()


	# def train_(self):
	#     test_data_path = os.path.join(self._dir_path,
	#                                   'train_user_item_rating.csv')
	#     lmdb_path = self._lmdb_path
	#     data_loader = DeepCoNNDataLoader(data_path=test_data_path,
	#                                      batch_size=self._batch_size,
	#                                      lmdb_path=lmdb_path,
	#                                      zero_index=self._zero_index,
	#                                      review_length=self._review_length,
	#                                      device=self._device)
	#     self._model.load_state_dict(torch.load(os.path.join(self._save_dir,
	#                                                         'NAECF.tar')))

	#     error = []
	#     pred = []
	#     uid = []
	#     iid = []
	#     rmse = []
	#     for users, items, ur, ir, r in data_loader:
	#         user_review_vec = \
	#             self._embedding(ur).unsqueeze(dim=1)
	#         item_review_vec = \
	#             self._embedding(ir).unsqueeze(dim=1)

	#         with torch.no_grad():
	#             batch_pred, _, _ = self._model(user_review_vec,
	#                                      item_review_vec)
	#             batch_error = batch_pred - r
	#         error.append(batch_error.cpu().numpy())
	#         pred.append(batch_pred.cpu().numpy())
	#         uid.append(users)
	#         iid.append(items)

	#     error_ = np.concatenate(error, axis=None) ** 2
	#     error = error_.mean().item()
	#     pred = np.concatenate(pred, axis=None)
	#     uid = np.concatenate(uid, axis=None)
	#     iid = np.concatenate(iid, axis=None)
	#     pred_df = pd.DataFrame(list(zip(uid, iid, pred)))

	#     pred_df.to_csv(os.path.join(self._res_dir, 'train_pred.csv'), index=None, header=['user', 'item', 'prediction'])

	#     error_df = pd.DataFrame(list(zip(uid, iid, error_)))
	#     error_df.to_csv(os.path.join(self._res_dir, 'train_error.csv'), index=None, header=['user', 'item', 'error'])
	#     # logger.info('Test MSE: {:.5f}'.format(error))
	#     # with open(os.path.join(self._save_dir, 'test_result.json'), 'w') as f:
	#     #     json.dump({'mse': error}, f)

	#     rmse_all = list(zip(uid, iid, error_))
	#     for i in range(self._user_size):
	#         if i not in uid:
	#             rmse.append([i, -1.0])
	#         else:
	#             x = np.array([list(x) for x in rmse_all if x[0]==i])
	#             rmse.append([i, np.sqrt(x[:, 2].sum()/len(x[:, 2]))])
	#     rmse = np.array(rmse)

	#     # rmse_df = pd.DataFrame(rmse)
	#     rmse_path = os.path.join(os.path.dirname(self._res_dir), 'train_rmse.csv')
	#     if not os.path.exists(os.path.dirname(rmse_path)):
	#         os.makedirs(os.path.join(os.path.dirname(rmse_path)))
	#     if os.path.exists(rmse_path):
	#         rmse_df = pd.read_csv(rmse_path, header=0, index_col=None, engine='python')
	#     else:
	#         rmse_df = pd.DataFrame()

	#     if 'user' not in rmse_df.columns:
	#         rmse_df['user'] = pd.Series(rmse[:, 0].astype(int))
	#     rmse_df[str(self._random_state)] = pd.Series(rmse[:, 1])
		
	#     # rmse_df.columns=col
	#     rmse_df.to_csv(os.path.join(rmse_path), header=True, index=0)


	#     data_loader.close_lmdb()

	


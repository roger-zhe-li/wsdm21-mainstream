# -*- coding: utf-8 -*-
import logging
from load_data import amazon_deepconn_load, amazon_fm_load, beer_deepconn_load, beer_fm_load
from model import DeepCoNNTrainTest, FMTrainTest
import os
import yaml
from guppy import hpy
import numpy as np
import torch
import argparse
import sys

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


parser = argparse.ArgumentParser(description='hyperparameters for DeepCoNN.')
parser.add_argument('--epochs', type=int, default=100,
					help='number of training epochs')
parser.add_argument('--batch_size', type=int, default=512,
					help='input batch size for training')
parser.add_argument('--lr', type=float, default=1e-5,
					help='learning rate')
parser.add_argument('--review_length', type=int, default=512,
					help='number of words for each user/item in the model')
parser.add_argument('--word_vector_dim', type=int, default=300,
					help='dimension of word2vec in the pretrained model')
parser.add_argument('--conv_length', type=int, default=3,
					help='length of convolutional layer in terms of words. e.g., 3 is for trigram')
parser.add_argument('--conv_kernel_num', type=int, default=100,
					help='number of convolutional kernels')
parser.add_argument('--fm_k', type=int, default=32,
					help='number of interaction latent factors for FM')
parser.add_argument('--latent_factor_num', type=int, default=100,
					help='number of user/item low-rank representation for recommendation')
parser.add_argument('--train_ratio', type=float, default=0.8,
					help='ratio for the training set')
parser.add_argument('--test_ratio', type=float, default=0.2,
					help='ratio for the test & valid set')
parser.add_argument('--seed', type=int, default=2020,
					help='value of random seed')
parser.add_argument('--no-cuda', action='store_true', default=False,
					help='enables CUDA training')
parser.add_argument('--data_store_path', nargs='?', default='./data/',
					help='Input data path.')
parser.add_argument('--dataset', type=str, default='music_instruments',
					choices=['music_instruments', 'instant_video', 'digital_music', 'beer'])
parser.add_argument('--rebuild', action='store_true', default=True,
					help='enables data rebuilding')
parser.add_argument('--coef', type=float, default=1.0,
					help='the tradeoff weight between text rebuilding and rating prediction')
parser.add_argument('--coef_u', type=float, default=1.0,
					help='the tradeoff weight between user text rebuilding and rating prediction')
parser.add_argument('--coef_i', type=float, default=1.0,
					help='the tradeoff weight between item text rebuilding and rating prediction')
parser.add_argument('--mode', type=int, default=1,
					choices=[0, 1, 2, 3, 4],
					help='different loss types')



args=parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


def set_random_seed(state=1):
	gens = (np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
	for set_state in gens:
		set_state(state)

RANDOM_STATE = args.seed
set_random_seed(RANDOM_STATE)


def deepconn():
	# with open('args_deepconn.yml', 'r', encoding='utf8') as f:
	#     args = yaml.load(f, Loader=yaml.FullLoader)
	# model_args = args['model']
	# training_args = args['training']
	# data_handle_args = args['data_handle']
	conv_length = args.conv_length
	dataset = args.dataset
	data_store_path = args.data_store_path
	latent_factor_num = args.latent_factor_num
	review_length = args.review_length
	word_vector_dim = args.word_vector_dim
	conv_kernel_num = args.conv_kernel_num
	fm_k = args.fm_k
	coef = args.coef
	coef_u = args.coef_u
	coef_i = args.coef_i
	mode = args.mode


	# dir_path = os.path.dirname(data_path)

	save_folder = 'latent_factor_' + str(latent_factor_num)

	if dataset == 'music_instruments':
		data_path = os.path.join(data_store_path, dataset, 'reviews_Musical_Instruments_5.json')
	elif dataset == 'instant_video':
		data_path = os.path.join(data_store_path, dataset, 'reviews_Amazon_Instant_Video_5.json')
	elif dataset == 'digital_music':
		data_path = os.path.join(data_store_path, dataset, 'reviews_Digital_Music_5.json')
	elif dataset == 'beer':
		data_path = os.path.join(data_store_path, dataset, 'beer_sampled.csv')
	else:
		print('dataset not available')

	dir_path = os.path.dirname(data_path)

	if dataset == 'beer':
	# 预处理数据
		beer_deepconn_load(data_path, 
							train_ratio=args.train_ratio,
							test_ratio=args.test_ratio,
							rebuild=args.rebuild,
							random_state=args.seed,
							coef=args.coef,
							coef_u=args.coef_u,
							coef_i=args.coef_i,
							mode=args.mode,
							dataset=args.dataset
							)
	elif dataset == 'music_instruments' or 'instant_video' or 'digital_music':
	# 预处理数据
		amazon_deepconn_load(data_path, 
							train_ratio=args.train_ratio,
							test_ratio=args.test_ratio,
							rebuild=args.rebuild,
							random_state=args.seed,
							coef=args.coef,
							coef_u=args.coef_u,
							coef_i=args.coef_i,
							mode=args.mode,
							dataset=args.dataset
							)

	# 训练模型
	train_test = DeepCoNNTrainTest(epoch=args.epochs,
								batch_size=args.batch_size,
								dir_path=dir_path,
								device=device,
								review_length=review_length,
								word_vector_dim=word_vector_dim,
								conv_length=conv_length,
								conv_kernel_num=conv_kernel_num,
								fm_k=fm_k,
								latent_factor_num=latent_factor_num,
								learning_rate=args.lr,
								save_folder=save_folder,
								random_state=args.seed,
								coef=args.coef,
								coef_u=args.coef_u,
								coef_i=args.coef_i,
								mode=args.mode,
								dataset=args.dataset)

	train_test.train()
	train_test.test(mode='test')
	train_test.valid(mode='valid')
	train_test.valid_test(mode='valid_test')
	train_test.train_valid(mode='train_valid')
	train_test.full(mode='full')
	train_test.train_(mode='train')
	train_test.del_db()


if __name__ == '__main__':
	# hp = hpy()
	deepconn()
	# print(hp.heap())
	# factorization_machine()

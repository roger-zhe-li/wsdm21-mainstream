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

logger = logging.getLogger('RATE.MF.train_test')


class DeepCoNNDataLoader:

    def __init__(self, data_path: str, batch_size, 
                 device: torch.device, shuffle=False):
        self._data = pd.read_csv(data_path)\
                         .reset_index(drop=True) \
                         .loc[:, ['user_id', 'item_id', 'rating']].to_numpy()
        # self._lmdb = lmdb.open(os.path.join(lmdb_path), readonly=True)
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
            rating = torch.from_numpy(rating).to(self._device)
                
            return users, items, rating

        else:
            self._index = 0
            raise StopIteration


class DataPreFetcher:

    def __init__(self, loader):
        self._loader = iter(loader)
        self._rating = None
        self.pre_load()

    def pre_load(self):
        try:
            self._user_review, self._item_review, self._rating \
                = next(self._loader)
        except StopIteration:
            self._rating = None
            return

    def next(self):
        # data = self._next_data
        rating = self._rating
        self.pre_load()
        return rating


class MF(nn.Module):

    def __init__(self, n_users, n_items, mf_k=10):
        """
        :param review_length: 评论单词数
        :param word_vec_dim: 词向量维度
        :param conv_length: 卷积核的长度
        :param conv_kernel_num: 卷积核数量
        :param latent_factor_num: 全连接输出的特征维度
        """
        super(MF, self).__init__()
        self.user_factors = torch.nn.Embedding(n_users, 
                                               mf_k
                                               )
        self.item_factors = torch.nn.Embedding(n_items, 
                                               mf_k
                                               )
    
    def forward(self, user, item):
        return (self.user_factors(user) * self.item_factors(item)).sum(1)

# noinspection PyUnreachableCode,PyArgumentList
class MFTrainTest:

    # noinspection PyUnresolvedReferences
    def __init__(self, epoch, batch_size, dir_path, device, 
                 mf_k,
                 learning_rate, save_folder, random_state, mode, dataset):
        """
        训练，测试DeepCoNN
        """
        self._epoch = epoch
        self._batch_size = batch_size
        
        self._dir_path = os.path.join(dir_path, 'NAECF_'+str(random_state))
        self._device = torch.device(device)
        self._save_dir = os.path.join(dir_path, 'MF_'+str(random_state), save_folder)        
        self._res_dir = os.path.join(dir_path, 'Results', 'latent_factor_'+str(mf_k), 'seed_'+str(random_state))
     
        
        self._random_state = random_state
        self._mode = mode
        self._dataset = dataset



        logger.info('epoch:{:<8d} batch size:{:d}'.format(epoch, batch_size))
        self.lmdb_dir = os.path.join('/tmp/zli6', self._dataset, 'MF_'+str(self._random_state))
        # if mode == 0:
        #     self._save_dir = os.path.join(dir_path, 'MF_'+str(random_state), save_folder, 'mode_0', str(coef))
        #     self._res_dir = os.path.join(dir_path, 'Results', 'latent_factor_'+str(latent_factor_num), 'mode_0', str(coef), 'seed_'+str(random_state))
        #     self._lmdb_path = os.path.join(self.lmdb_dir, 'mode_0', 'lmdb_'+str(self._coef))
        # if mode == 1:
        #     self._save_dir = os.path.join(dir_path, 'MF_'+str(random_state), save_folder, 'mode_1', str(coef))
        #     self._res_dir = os.path.join(dir_path, 'Results', 'latent_factor_'+str(latent_factor_num), 'mode_1', str(coef), 'seed_'+str(random_state))
        #     self._lmdb_path = os.path.join(self.lmdb_dir, 'mode_1', 'lmdb_'+str(self._coef))
        # if mode == 2:
        #     self._save_dir = os.path.join(dir_path, 'MF_'+str(random_state), save_folder, 'mode_2', 'user_'+str(coef_u)+'_item_'+str(coef_i))
        #     self._res_dir = os.path.join(dir_path, 'Results', 'latent_factor_'+str(latent_factor_num), 'mode_2', 'user_'+str(coef_u)+'_item_'+str(coef_i), 'seed_'+str(random_state))
        #     self._lmdb_path = os.path.join(self.lmdb_dir, 'mode_2', 'lmdb_user_'+str(self._coef_u)+'_item_'+str(self._coef_i))
        # if mode == 3: 
        #     self._save_dir = os.path.join(dir_path, 'MF_'+str(random_state), save_folder, 'mode_3', 'user_'+str(coef_u)+'_item_'+str(coef_i))
        #     self._res_dir = os.path.join(dir_path, 'Results', 'latent_factor_'+str(latent_factor_num), 'mode_3', 'user_'+str(coef_u)+'_item_'+str(coef_i), 'seed_'+str(random_state))
        #     self._lmdb_path = os.path.join(self.lmdb_dir, 'mode_3', 'lmdb_user_'+str(self._coef_u)+'_item_'+str(self._coef_i))
        if mode == 5: 
            self._save_dir = os.path.join(dir_path, 'MF_'+str(random_state), save_folder, 'mode_5_')
            self._res_dir = os.path.join(dir_path, 'Results', 'latent_factor_'+str(mf_k), 'mode_5_', 'seed_'+str(random_state))
            self._lmdb_path = os.path.join(self.lmdb_dir, 'mode_5_')


        # print(self._lmdb_path)
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir)
        if not os.path.exists(self._res_dir):
            os.makedirs(self._res_dir)

        # read data
        self._train_data = pd.read_csv(
            os.path.join(self._dir_path, 'train_user_item_rating.csv'))
        self._test_data = pd.read_csv(
            os.path.join(self._dir_path, 'test_user_item_rating.csv'))

        with open(os.path.join(self._dir_path, 'dataset_meta_info.json'),
                  'r') as f:
            dataset_meta_info = json.load(f)
        self._user_size = dataset_meta_info['user_size']
        self._item_size = dataset_meta_info['item_size']
        self._dataset_size = dataset_meta_info['dataset_size']

        # initial DeepCoNN model
        self._model = MF(self._user_size, self._item_size, mf_k).to(self._device)

        self._optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self._model.parameters()),
                                           lr=learning_rate)
        # self._loss_func = torch.nn.MSELoss()
        self._loss_func = torch.nn.MSELoss(reduction='none')

        # user and item count
        user_ids = self._train_data.user_id.tolist()
        user_count = []
        for i in range(self._user_size):
            user_count.append(user_ids.count(i))

        item_ids = self._train_data.item_id.tolist()
        item_count = []
        for j in range(self._item_size):
            item_count.append(item_ids.count(j))


        # load pretrained embedding
        logger.info('Initialize word embedding model for pytorch')
        logger.info('Model initialized, start training...')
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
                                               
                                               device=self._device,
                                               shuffle=True)

        valid_data_loader = DeepCoNNDataLoader(data_path=valid_data_path,
                                               batch_size=self._batch_size,
                                               
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
            for users, items, rating \
                    in train_data_loader:

                users = torch.LongTensor(users).to(self._device)
                items = torch.LongTensor(items).to(self._device)
                pred = self._model(users, items)
                
                if self._mode == 5:
                    train_loss = self._loss_func(pred, rating.flatten())
                    
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
            # print(np.mean(train_loss_value))

            error = []
            for user, item, valid_rating \
                    in valid_data_loader:
                user = torch.LongTensor(user).to(self._device)
                item = torch.LongTensor(item).to(self._device)
                with torch.no_grad():
                    batch_pred = self._model(user,
                                             item)

                    batch_error = batch_pred - valid_rating
                    error.append(batch_error.cpu().numpy())

            error = np.concatenate(error, axis=None)**2
            error = error.mean().item()

            if best_valid_loss > error:
                best_model_state_dict = copy.deepcopy(self._model.state_dict())
                best_valid_loss = error
                best_valid_epoch = e
            # if e == self._epoch - 1:
            #     best_model_state_dict = copy.deepcopy(self._model.state_dict())
            #     valid_loss = error
            #     best_valid_epoch = e

                torch.save(best_model_state_dict,
                           os.path.join(self._save_dir,
                                        'MF.tar'))
                # print(os.path.join(self._save_dir, 'MF.tar'))
            logger.info(
                'epoch: {}, train loss: {:.5f}, valid mse_loss: {:.5f}'
                .format(e, np.mean(train_loss_value), error))

        with open(os.path.join(self._save_dir, 'training.json'), 'w') as f:
            json.dump({'seed': self._random_state,
                        'epoch': best_valid_epoch,
                       'valid_loss': np.sqrt(best_valid_loss)},
                      f)

        # train_data_loader.close_lmdb()
        # valid_data_loader.close_lmdb()


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
                                         # lmdb_path=lmdb_path,
                                         # zero_index=self._zero_index,
                                         # review_length=self._review_length,
                                         device=self._device)
        self._model.load_state_dict(torch.load(os.path.join(self._save_dir,
                                                            'MF.tar')))

        error = []
        pred = []
        uid = []
        iid = []
        rmse = []

        for users, items, r in data_loader:
            # user_review_vec = \
            #     self._embedding(ur).unsqueeze(dim=1)
            # item_review_vec = \
            #     self._embedding(ir).unsqueeze(dim=1)
            users = torch.LongTensor(users).to(self._device)
            items = torch.LongTensor(items).to(self._device)

            with torch.no_grad():
                batch_pred = self._model(users, items)
                # clamp with boundings
                batch_pred = torch.clamp(batch_pred, 1, 5)

                batch_error = batch_pred - r
            error.append(batch_error.cpu().numpy())
            pred.append(batch_pred.cpu().numpy())
            uid.append(users.cpu().numpy())
            iid.append(items.cpu().numpy())

        error_ = np.concatenate(error, axis=None) ** 2
        error = error_.mean().item()
        pred = np.concatenate(pred, axis=None)
        uid = np.concatenate(uid, axis=None)
        iid = np.concatenate(iid, axis=None)
        pred_df = pd.DataFrame(list(zip(uid, iid, pred)))
        pred_df.to_csv(os.path.join(self._res_dir, mode+'_pred_bounded.csv'), index=None, header=['user', 'item', 'prediction'])

        error_df = pd.DataFrame(list(zip(uid, iid, error_)))
        error_df.to_csv(os.path.join(self._res_dir, mode+'_error_bounded.csv'), index=None, header=['user', 'item', 'error'])

        rmse_all = list(zip(uid, iid, error_))
        uid_unique = np.unique(uid)
        print(len(rmse_all))
        print(self._user_size)
        for i in range(self._user_size):
            if i not in uid_unique:
                rmse.append([int(i), -1.0])
            else:
                x = np.array([list(x) for x in rmse_all if x[0]==i])
                rmse.append([int(i), np.sqrt(x[:, 2].sum()/len(x[:, 2]))])
        rmse = np.array(rmse)
        print(rmse.shape)

        # rmse_df = pd.DataFrame(rmse)
        rmse_path = os.path.join(os.path.dirname(self._res_dir), mode+'_rmse_bounded.csv')
        if not os.path.exists(os.path.dirname(rmse_path)):
            os.makedirs(os.path.join(os.path.dirname(rmse_path)))
        if os.path.exists(rmse_path):
            rmse_df = pd.read_csv(rmse_path, header=0, index_col=None, engine='python', error_bad_lines=False)
        else:
            rmse_df = pd.DataFrame()

        # if 'user' not in rmse_df.columns:
        rmse_df['user'] = pd.Series(rmse[:, 0].astype(int))
        rmse_df[str(self._random_state)] = pd.Series(rmse[:, 1])

        # print(os.path.abspath(rmse_path))
        
        # rmse_df.columns=col
        rmse_df.to_csv(rmse_path, header=True, index=0)
        
        if mode == 'test':
            logger.info('Test MSE: {:.5f}'.format(error))
            with open(os.path.join(self._save_dir, 'test_result.json'), 'w') as f:
                json.dump({'rmse': np.sqrt(error),
                           'seed': self._random_state}, 
                        f)

        # data_loader.close_lmdb()


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

    # def test(self):
    #     test_data_path = os.path.join(self._dir_path,
    #                                   'test_user_item_rating.csv')
    #     lmdb_path = self._lmdb_path
    #     data_loader = DeepCoNNDataLoader(data_path=test_data_path,
    #                                      batch_size=self._batch_size,
                                         
    #                                      device=self._device)
    #     self._model.load_state_dict(torch.load(os.path.join(self._save_dir,
    #                                                         'MF.tar')))

    #     error = []
    #     pred = []
    #     uid = []
    #     iid = []
    #     rmse = []
    #     for users, items, rating \
    #                 in data_loader:

    #         users = torch.LongTensor(users).to(self._device)
    #         items = torch.LongTensor(items).to(self._device)
    #         batch_pred = self._model(users, items)

    #         with torch.no_grad():
    #             batch_pred = self._model(users,
    #                                      items)
    #             batch_error = batch_pred - rating
            
    #         error.append(batch_error.cpu().numpy())
    #         pred.append(batch_pred.cpu().numpy())
    #         uid.append(users.cpu().numpy())
    #         iid.append(items.cpu().numpy())

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


    # def valid(self):
    #     test_data_path = os.path.join(self._dir_path,
    #                                   'valid_user_item_rating.csv')
    #     lmdb_path = self._lmdb_path
    #     data_loader = DeepCoNNDataLoader(data_path=test_data_path,
    #                                      batch_size=self._batch_size,
                                        
    #                                      device=self._device)
    #     self._model.load_state_dict(torch.load(os.path.join(self._save_dir,
    #                                                         'MF.tar')))

    #     error = []
    #     pred = []
    #     uid = []
    #     iid = []
    #     rmse = []
    #     for users, items, rating in data_loader:

    #         users = torch.LongTensor(users).to(self._device)
    #         items = torch.LongTensor(items).to(self._device)
    #         batch_pred = self._model(users, items)

    #         with torch.no_grad():
    #             batch_pred = self._model(users,
    #                                      items)
    #             batch_error = batch_pred - rating
    #         error.append(batch_error.cpu().numpy())
    #         pred.append(batch_pred.cpu().numpy())
    #         uid.append(users.cpu().numpy())
    #         iid.append(items.cpu().numpy())

    #     error_ = np.concatenate(error, axis=None) ** 2
    #     error = error_.mean().item()
    #     pred = np.concatenate(pred, axis=None)
    #     uid = np.concatenate(uid, axis=None)
    #     iid = np.concatenate(iid, axis=None)
    #     pred_df = pd.DataFrame(list(zip(uid, iid, pred)))

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




    # def valid_test(self):
    #     test_data_path = os.path.join(self._dir_path,
    #                                   'valid_test_user_item_rating.csv')
    #     lmdb_path = self._lmdb_path
    #     data_loader = DeepCoNNDataLoader(data_path=test_data_path,
    #                                      batch_size=self._batch_size,
                                         
    #                                      device=self._device)
    #     self._model.load_state_dict(torch.load(os.path.join(self._save_dir,
    #                                                         'MF.tar')))

    #     error = []
    #     pred = []
    #     uid = []
    #     iid = []
    #     rmse = []
    #     for users, items, rating \
    #                 in data_loader:

    #         users = torch.LongTensor(users).to(self._device)
    #         items = torch.LongTensor(items).to(self._device)
    #         batch_pred = self._model(users, items)

    #         with torch.no_grad():
    #             batch_pred = self._model(users,
    #                                      items)
    #             batch_error = batch_pred - rating
    #         error.append(batch_error.cpu().numpy())
    #         pred.append(batch_pred.cpu().numpy())
    #         uid.append(users.cpu().numpy())
    #         iid.append(items.cpu().numpy())

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


    #     # data_loader.close_lmdb()


    # def train_valid(self):
    #     test_data_path = os.path.join(self._dir_path,
    #                                   'train_valid_user_item_rating.csv')
    #     lmdb_path = self._lmdb_path
    #     data_loader = DeepCoNNDataLoader(data_path=test_data_path,
    #                                      batch_size=self._batch_size,
                                         
    #                                      device=self._device)
    #     self._model.load_state_dict(torch.load(os.path.join(self._save_dir,
    #                                                         'MF.tar')))

    #     error = []
    #     pred = []
    #     uid = []
    #     iid = []
    #     rmse = []
    #     for users, items, rating \
    #                 in data_loader:

    #         users = torch.LongTensor(users).to(self._device)
    #         items = torch.LongTensor(items).to(self._device)
    #         batch_pred = self._model(users, items)

    #         with torch.no_grad():
    #             batch_pred = self._model(users,
    #                                      items)
    #             batch_error = batch_pred - rating
    #         error.append(batch_error.cpu().numpy())
    #         pred.append(batch_pred.cpu().numpy())
    #         uid.append(users.cpu().numpy())
    #         iid.append(items.cpu().numpy())

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


    # def full(self):
    #     test_data_path = os.path.join(self._dir_path,
    #                                   'full_rating.csv')
    #     lmdb_path = self._lmdb_path
    #     data_loader = DeepCoNNDataLoader(data_path=test_data_path,
    #                                      batch_size=self._batch_size,
                                         
    #                                      device=self._device)
    #     self._model.load_state_dict(torch.load(os.path.join(self._save_dir,
    #                                                         'MF.tar')))

    #     error = []
    #     pred = []
    #     uid = []
    #     iid = []
    #     rmse = []
    #     for users, items, rating \
    #                 in data_loader:

    #         users = torch.LongTensor(users).to(self._device)
    #         items = torch.LongTensor(items).to(self._device)
    #         batch_pred = self._model(users, items)

    #         with torch.no_grad():
    #             batch_pred = self._model(users,
    #                                      items)
    #             batch_error = batch_pred - rating
    #         error.append(batch_error.cpu().numpy())
    #         pred.append(batch_pred.cpu().numpy())
    #         uid.append(users.cpu().numpy())
    #         iid.append(items.cpu().numpy())

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


    # def train_(self):
    #     test_data_path = os.path.join(self._dir_path,
    #                                   'train_user_item_rating.csv')
    #     lmdb_path = self._lmdb_path
    #     data_loader = DeepCoNNDataLoader(data_path=test_data_path,
    #                                      batch_size=self._batch_size,
                                         
    #                                      device=self._device)
    #     self._model.load_state_dict(torch.load(os.path.join(self._save_dir,
    #                                                         'MF.tar')))

    #     error = []
    #     pred = []
    #     uid = []
    #     iid = []
    #     rmse = []
    #     for users, items, rating \
    #                 in data_loader:

    #         users = torch.LongTensor(users).to(self._device)
    #         items = torch.LongTensor(items).to(self._device)
    #         batch_pred = self._model(users, items)

    #         with torch.no_grad():
    #             batch_pred = self._model(users,
    #                                      items)
    #             batch_error = batch_pred - rating
    #         error.append(batch_error.cpu().numpy())
    #         pred.append(batch_pred.cpu().numpy())
    #         uid.append(users.cpu().numpy())
    #         iid.append(items.cpu().numpy())

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


        # data_loader.close_lmdb()

    def del_db(self):
        lmdb_path = self._lmdb_path
        if os.path.exists(os.path.join(lmdb_path, 'data.mdb')):
            os.remove(os.path.join(lmdb_path, 'data.mdb'))
            os.remove(os.path.join(lmdb_path, 'lock.mdb'))


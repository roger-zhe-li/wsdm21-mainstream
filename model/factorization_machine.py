# -*- coding: utf-8 -*-
import json
import math
import os
import random

import lmdb
import msgpack
from tqdm import tqdm
import copy
from torch.utils.data import TensorDataset, DataLoader
from util import data_split_pandas
import logging
import torch.nn as nn
import torch
import pandas as pd
import numpy as np

logger = logging.getLogger('RATE.FM.train_test')


class FactorizationMachine(nn.Module):

    # noinspection PyArgumentList
    def __init__(self, sample_size: int, factor_size: int):
        super(FactorizationMachine, self).__init__()
        self._linear = nn.Linear(sample_size, 1)
        self._v = torch.nn.Parameter(torch.normal(0, .001,
                                                  (sample_size, factor_size)))
        self._drop = nn.Dropout(0.2)

    def forward(self, x):
        # linear regression
        w = self._linear(x).squeeze()

        # cross feature
        inter1 = torch.matmul(x, self._v)
        inter2 = torch.matmul(x**2, self._v**2)
        inter = (inter1**2 - inter2) * 0.5
        inter = self._drop(inter)
        inter = torch.sum(inter, dim=1)

        return w + inter


class FMDataLoader:

    def __init__(self, lmdb_path, feature_size, batch_size,
                 device: torch.device, shuffle=False):
        self._lmdb_data = lmdb.open(lmdb_path, readonly=True)
        lmdb_entry_size = self._lmdb_data.stat()['entries']
        self._feature_size = feature_size
        self._batch_size = batch_size
        self._device = device

        self._idx_list = list(range(lmdb_entry_size))
        self._shuffle = shuffle
        if shuffle:
            random.shuffle(self._idx_list)
        self._idx = 0

    def __len__(self):
        return math.ceil(len(self._idx_list) // self._batch_size)

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx < len(self._idx_list):
            idx_ls = self._idx_list[self._idx: self._idx+self._batch_size]
            batch_size = len(idx_ls)
            self._idx += self._batch_size

            with self._lmdb_data.begin() as txn:
                entry_data = [msgpack.unpackb(txn.get(str(x).encode()),
                                              raw=False)
                              for x in idx_ls]

            x_index = [x['put_idx'] for x in entry_data]
            x_value = [x['put_value'] for x in entry_data]
            rating = [x['rating'] for x in entry_data]

            if self._device == torch.device('cpu'):
                x_tensor = np.zeros((batch_size, self._feature_size),
                                    dtype=np.float32)
                x_tensor = torch.from_numpy(x_tensor)
            else:
                x_tensor = torch.cuda.FloatTensor(batch_size,
                                                  self._feature_size,
                                                  device=self._device).fill_(0)

            x_index = map(lambda x: torch.from_numpy(np.asarray(x, np.longlong)).to(self._device),
                          x_index)
            x_value = list(
                map(lambda x: torch.from_numpy(np.asarray(x, np.float32)).to(self._device),
                    x_value))

            for i, (x_i, x_v) in enumerate(zip(x_index, x_value)):
                x_tensor[i].scatter_(dim=0, index=x_i, src=x_v)

            rating = np.asarray(rating, dtype=np.float32)
            rating = torch.from_numpy(rating).to(self._device)

            return x_tensor, rating
        else:
            self._idx = 0
            if self._shuffle:
                random.shuffle(self._idx_list)
            raise StopIteration

    def close_lmdb(self):
        self._lmdb_data.close()


class FMTrainTest:

    def __init__(self, epoch, batch_size, dir_path, device, model_args,
                 learning_rate, save_folder):
        self._epoch = epoch
        self._batch_size = batch_size
        self._dir_path = os.path.join(dir_path, 'FM')
        self._device = torch.device(device)
        self._model_args = model_args
        self._lr = learning_rate
        self._save_dir = os.path.join(self._dir_path, save_folder)

        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir)

        with open(os.path.join(self._dir_path, 'dataset_meta_info.json'),
                  'r') as f:
            dataset_meta_info = json.load(f)

        self._user_size = dataset_meta_info['user_size']
        self._item_size = dataset_meta_info['item_size']
        self._feature_size = self._user_size + 3 * self._item_size + 1

        self._model = FactorizationMachine(self._user_size+self._item_size*3+1,
                                           model_args,).to(self._device)

        self._optimizer = torch.optim.Adam(self._model.parameters(),
                                           lr=learning_rate)
        self._loss_func = torch.nn.MSELoss()

        self._test_data = FMDataLoader(os.path.join(self._dir_path, 'test_lmdb'),
                                       self._feature_size,
                                       self._batch_size,
                                       self._device,
                                       shuffle=False)

    def train(self):
        # data = pd.read_json(os.path.join(self._dir_path,
        #                                  'train_user_item_rating.json'))
        # train_data, valid_data = data_split_pandas(data, 0.9, 0.1)

        train_data = FMDataLoader(os.path.join(self._dir_path, 'train_lmdb'),
                                  self._feature_size,
                                  self._batch_size,
                                  self._device,
                                  shuffle=True)

        valid_data = FMDataLoader(os.path.join(self._dir_path, 'valid_lmdb'),
                                  self._feature_size,
                                  self._batch_size,
                                  self._device,
                                  shuffle=False)

        logger.info('Start training.')
        best_valid_loss = float('inf')
        best_valid_epoch = 0
        for e in range(self._epoch):
            train_loss = None
            for x, y in tqdm(train_data):
                pred = self._model(x)
                train_loss = self._loss_func(pred, y.flatten())
                self._optimizer.zero_grad()
                train_loss.backward()
                self._optimizer.step()

            # valid
            # error = torch.cuda.FloatTensor([0], device=self._device)
            error = []
            with torch.no_grad():
                for x, y in tqdm(valid_data):
                    pred = self._model(x)
                    batch_error = (pred - y.flatten())

                    error.append(batch_error.cpu().numpy())

            error = np.concatenate(error, axis=None)**2
            error = error.mean()
            if best_valid_loss > error:
                best_valid_loss = error
                best_valid_epoch = e

                # save
                torch.save(self._model.state_dict(),
                           os.path.join(self._save_dir,
                                        'FM.tar'))
            logger.info(
                'epoch: {:<3d} train mse_loss: {:.5f}, valid mse_loss: {:.5f}'
                .format(e, train_loss, error))

        with open(os.path.join(self._save_dir, 'training.json'), 'w') as f:
            json.dump({'epoch': best_valid_epoch,
                       'valid_loss': best_valid_loss.item()},
                      f)

    def test(self):
        self._model.load_state_dict(torch.load(os.path.join(self._save_dir,
                                                            'FM.tar')))

        # error = torch.FloatTensor().to(self._device)
        # pred = torch.FloatTensor().to(self._device)
        error = []
        for x, y in self._test_data:
            with torch.no_grad():
                batch_pred = self._model(x)
                # pred = torch.cat((pred, batch_pred))

                batch_error = (batch_pred - y.flatten())
                error.append(batch_error.cpu().numpy())

        error = np.concatenate(error, axis=None)**2
        error = error.mean()

        logger.info('Test MSE: {:.5f}'.format(error))

        with open(os.path.join(self._save_dir, 'test_result.json'), 'w') as f:
            json.dump({'mse': error.item()}, f)

        # data['predict'] = pred.tolist()
        # data.to_json(os.path.join(self._save_dir,
        #                           'test_result_detail.json'))

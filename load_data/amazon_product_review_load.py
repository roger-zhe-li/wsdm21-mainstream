# -*- coding: utf-8 -*-
from datetime import datetime
import json
from tqdm import tqdm
import numpy as np
import logging
import os
import shutil
import lmdb
import pandas as pd
from util import data_split_pandas
from util import split_sentence_to_word_list, words2ids
import msgpack

logger = logging.getLogger('RATE.load_data')


def load_data(data_path):
    logger.info('Start reading data to pandas.')
    data_pd = pd.read_json(data_path, lines=True)
    data_pd = data_pd.rename(index=int, columns={'asin': 'item',
                                                 'overall': 'rating',
                                                 'reviewText': 'review_text',
                                                 'reviewerID': 'user',
                                                 'unixReviewTime': 'time'})
    data_pd['review_id'] = data_pd.index

    del data_pd['helpful']
    del data_pd['summary']

    return data_pd


def load_data_deepconn(data_path, train_ratio=.8, test_ratio=.2, rebuild=False, random_state=2020, coef=0.0,
                        coef_u=1.0, coef_i=1.0, mode=2, dataset='music_instruments'):
    """
    数据预处理, 并将review信息存入以lmdb保存在 data_path/../DeepCoNN/lmdb/
    保存至本地文件：数据集基本信息,
    train_user-item-rating-review_id, test_user-item-rating-review_id,
    :param data_path: 源数据路径
    :param train_ratio:
    :param test_ratio:
    :param rebuild:
    :return:
    """

    folder = os.path.dirname(data_path)
    folder = os.path.join(folder, 'NAECF_'+str(random_state))

    if not rebuild:
        if os.path.exists(os.path.join(folder, 'dataset_meta_info.json')):
            return

    if not os.path.exists(folder):
        os.makedirs(folder)

    data_pd = load_data(data_path)

    user_ids, data_pd = get_unique_id(data_pd, 'user')
    item_ids, data_pd = get_unique_id(data_pd, 'item')

    user_size = len(user_ids)
    item_size = len(item_ids)
    dataset_size = len(data_pd)

    dataset_meta_info = {'dataset_size': dataset_size,
                         'user_size': user_size,
                         'item_size': item_size}

    logger.info('convert review sentences into tokens and token ids')
    review = data_pd['review_text']
    # print(type(review))
    review_tokens = review.apply(lambda x: split_sentence_to_word_list(x))
    data_pd['review_token_ids'] = review_tokens.apply(lambda x: words2ids(x))
    # review = review_tokens.combine(review_token_ids,
    #                                lambda x, y: json.dumps({'tokens': x,
    #                                                         'token_ids': y}))
    # review = review.apply(lambda x: x.encode())

    user_review = data_pd.groupby('user_id')['review_token_ids'] \
        .apply(lambda x: sum(x, []))
    item_review = data_pd.groupby('item_id')['review_token_ids'] \
        .apply(lambda x: sum(x, []))

    # user_review = user_review.rename(index=str)
    user_review.index = user_review.index.map(lambda x: 'user-' + str(x))
    item_review.index = item_review.index.map(lambda x: 'item-' + str(x))

    concat_review = pd.concat([user_review, item_review])

    mean_review_length = concat_review.map(len).mean()
    dataset_meta_info['mean_review_length'] = int(mean_review_length)

    review_memory = concat_review.memory_usage(index=True, deep=True)
    review_memory = int(review_memory * 1.5)
    review = concat_review.to_dict()

    logger.info('Save data to lmdb')
    if mode == 0:
        lmdb_path = os.path.join('/home/nfs/zli6', dataset, 'NAECF_'+str(random_state), 'mode_0', 'lmdb_'+str(coef))
    elif mode == 1:
        lmdb_path = os.path.join('/home/nfs/zli6', dataset, 'NAECF_'+str(random_state), 'mode_1', 'lmdb_'+str(coef))
    elif mode == 2:
        lmdb_path = os.path.join('/home/nfs/zli6', dataset, 'NAECF_'+str(random_state), 'mode_2', 'lmdb_user_'+str(coef_u)+'_item_'+str(coef_i))
    elif mode == 3:
        lmdb_path = os.path.join('/home/nfs/zli6', dataset, 'NAECF_'+str(random_state), 'mode_3', 'lmdb_user_'+str(coef_u)+'_item_'+str(coef_i))
    elif mode == 4:
        lmdb_path = os.path.join('/home/nfs/zli6', dataset, 'NAECF_'+str(random_state), 'mode_4', 'lmdb_user_'+str(coef_u)+'_item_'+str(coef_i))
    elif mode == 5:
        lmdb_path = os.path.join('/tmp/zli6', dataset, 'MF_'+str(random_state), 'mode_5_')
    if os.path.exists(lmdb_path):
        shutil.rmtree(lmdb_path)
    if not os.path.exists(lmdb_path):
        os.makedirs(lmdb_path)

    env = lmdb.open(lmdb_path, map_size=review_memory)
    with env.begin(write=True) as txn:
        for k, v in review.items():
            txn.put(str(k).encode(), msgpack.packb(v))

    data_pd = data_pd.loc[:, ['user', 'user_id', 'item', 'item_id', 'rating']]
    logger.info('Split training and test dataset')
    train_uir, valid_test_uir = data_split_pandas(data_pd, train_ratio, test_ratio, random_state)
    test_uir, valid_uir = data_split_pandas(valid_test_uir, 0.5, 0.5, random_state)
    train_valid_uir = pd.concat([train_uir, valid_uir])
    data_pd.to_csv(
        os.path.join(folder, 'full_rating.csv'))
    train_uir.to_csv(
        os.path.join(folder, 'train_user_item_rating.csv'))
    test_uir.to_csv(
        os.path.join(folder, 'test_user_item_rating.csv'))
    valid_uir.to_csv(
        os.path.join(folder, 'valid_user_item_rating.csv'))
    valid_test_uir.to_csv(
        os.path.join(folder, 'valid_test_user_item_rating.csv'))
    train_valid_uir.to_csv(
        os.path.join(folder, 'train_valid_user_item_rating.csv'))

    with open(os.path.join(folder, 'dataset_meta_info.json'), 'w') as f:
        json.dump(dataset_meta_info, f)

    logger.info('Load data finished.')


def load_data_fm(data_path, train_ratio, test_ratio, rebuild):
    """
    保存以下信息：数据集基本信息
    :param data_path:
    :param train_ratio:
    :param test_ratio:
    :param rebuild:
    :return:
    """

    folder = os.path.dirname(data_path)
    folder = os.path.join(folder, 'FM')
    if not rebuild:
        if os.path.exists(os.path.join(folder, 'dataset_meta_info.json')):
            return

    data_pd = load_data(data_path)

    del data_pd['review_text']

    if not os.path.exists(folder):
        os.makedirs(folder)

    user_ids, data_pd = get_unique_id(data_pd, 'user')
    item_ids, data_pd = get_unique_id(data_pd, 'item')

    user_size = len(user_ids)
    item_size = len(item_ids)
    dataset_size = len(data_pd)
    min_date = data_pd['time'].min().item()
    dataset_meta_info = {'dataset_size': dataset_size,
                         'user_size': user_size,
                         'item_size': item_size,
                         'min_date': min_date}
    min_date = datetime.fromtimestamp(min_date)

    logger.info('Get <user, item, rating> triplet.')
    uir = data_pd.loc[:, ['user', 'user_id', 'item',
                          'item_id', 'time', 'rating']]
    logger.info('Split data into train, valid and test sets')

    # get users' rated items
    tqdm.pandas(desc="get users' rated items")
    uir['rated_item'] = uir.progress_apply(
        lambda r: (r['time'], r['item_id']), axis='columns')

    tqdm.pandas(desc='Group by user to get rated items')
    user_group = uir.groupby('user_id')['rated_item'].progress_apply(list)

    tqdm.pandas(desc='Sort rated items')
    user_group = user_group.progress_apply(
        lambda r: sorted(r, key=lambda l: l[0]))

    del uir['rated_item']
    logger.info('Merge rated items into main data')
    uir = pd.merge(left=uir,
                   left_on='user_id',
                   right=user_group,
                   right_index=True,
                   how='inner')

    tqdm.pandas(desc='Filter un-rated items 1')
    uir['rated_item'] = uir.progress_apply(
        lambda r: list(filter(lambda l: l[0] > r['time'], r['rated_item'])),
        axis='columns')

    tqdm.pandas(desc='Filter un-rated items 2')
    uir['rated_item'] = uir.progress_apply(
        lambda r: [l[1] for l in r['rated_item']],
        axis='columns')

    tqdm.pandas(desc='Calculate month')
    uir['month'] = uir['time'].progress_apply(
        lambda x: (datetime.fromtimestamp(x) - min_date).days // 30)

    tqdm.pandas(desc='Get put_idx, put_value')
    uir = uir.progress_apply(
        lambda x: get_put_idx_value(x, user_size, item_size),
        axis='columns')

    train_uir, test_uir = data_split_pandas(uir, train_ratio, test_ratio)
    valid_uir, test_uir = data_split_pandas(test_uir, 0.5, 0.5)

    with open(os.path.join(folder, 'dataset_meta_info.json'), 'w') as f:
        json.dump(dataset_meta_info, f)

    save_pandas_to_lmdb(train_uir, os.path.join(folder, 'train_lmdb'))
    save_pandas_to_lmdb(valid_uir, os.path.join(folder, 'valid_lmdb'))
    save_pandas_to_lmdb(test_uir, os.path.join(folder, 'test_lmdb'))

    logger.info('Load data finished.')


def save_pandas_to_lmdb(df: pd.DataFrame, lmdb_path: str, max_size: int = None):
    if max_size is None:
        max_size = df.memory_usage(deep=True).sum() * 1.5
        max_size = int(max_size)

    df.reset_index(drop=True)
    df = df.to_dict(orient='index')

    env = lmdb.open(lmdb_path, map_size=max_size)
    with env.begin(write=True) as txn:
        for k, v in tqdm(df.items()):
            txn.put(str(k).encode(), msgpack.packb(v))


def dataset_statistic(data_frame: pd.DataFrame, folder: str):
    """
    statistics of data: #user, #item, #review, #word,
    #review per user, #word per user
    :param data_frame: columns: user, item, review_text
    :param folder: fig save folder path
    :return: None
    """
    data_frame['review_word_count'] = \
        data_frame.apply(lambda x: len(x['review_text'].split()), axis=1)
    user_size = data_frame['user'].drop_duplicates().size
    item_size = data_frame['item'].drop_duplicates().size
    review_size = data_frame.size
    word_count = data_frame['review_word_count'].sum()

    attr_list = ('#user', '#item', '#review', '#words',
                 '#review per user', '#words per user')
    rows_format = '{:<10}' * 4 + '{:<20}' * 2
    print(rows_format.format(*attr_list))
    dash = '-' * 40
    print(dash)
    print(rows_format.format(user_size,
                             item_size,
                             review_size,
                             word_count,
                             word_count / user_size,
                             word_count / review_size))


def get_unique_id(data_pd: pd.DataFrame, column: str) -> (dict, pd.DataFrame):
    """
    获取指定列的唯一id
    :param data_pd: pd.DataFrame 数据
    :param column: 指定列
    :return: dict: {value: id}
    """
    new_column = '{}_id'.format(column)
    assert new_column not in data_pd.columns
    temp = data_pd.loc[:, [column]].drop_duplicates().reset_index(drop=True)
    temp[new_column] = temp.index
    temp.index = temp[column]
    del temp[column]
    # data_pd.merge()
    data_pd = pd.merge(left=data_pd,
                       right=temp,
                       left_on=column,
                       right_index=True,
                       how='left')

    return temp[new_column].to_dict(), data_pd


def get_put_idx_value(data, user_size, item_size):
    user_id = data['user_id']
    item_id = data['item_id']
    rating = data['rating']
    month = data['month']
    rated_items = data['rated_item']
    x_length = user_size + 3 * item_size + 1

    x = np.zeros(x_length)

    # user_id, item_id one-hot
    x_idx = [user_id, user_id + item_id]
    x_value = [1, 1]

    if len(rated_items):
        idx_temp = user_size + item_size
        x_idx += [i + idx_temp for i in rated_items]
        x_value += [1 / len(rated_items)] * len(rated_items)

    idx_temp = user_size + item_size * 2 + 1
    x_idx.append(idx_temp)
    x_value.append(month)

    # last rated movie
    if len(rated_items):
        x_idx.append(rated_items[-1])
        x_value.append(1)

    data['put_idx'] = x_idx
    data['put_value'] = x_value

    return data

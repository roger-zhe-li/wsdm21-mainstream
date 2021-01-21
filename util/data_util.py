# -*- coding: utf-8 -*-
import pandas as pd


def data_split_pandas(data: pd.DataFrame, train_ratio, test_ratio, random_state):
    assert (train_ratio + test_ratio) == 1., 'ratio sum != 1'
    data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)  # shuffle
    train_index = int(len(data) * train_ratio)

    train_data = data.loc[:train_index, :].reset_index(drop=True)
    test_data = data.loc[train_index:, :].reset_index(drop=True)

    return train_data, test_data

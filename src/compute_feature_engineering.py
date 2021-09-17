#!/usr/bin/env python3
#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202108311525
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
本模块(compute_feature_engineering.py)针对KPI数据进行特征工程。

@References:
----------
[1] https://www.kaggle.com/lucasmorin/running-algos-fe-for-fast-inference
[2] https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
[3] Rakthanmanon, Thanawin, et al. "Searching and mining trillions of time series subsequences under dynamic time warping." Proceedings of the 18th ACM SIGKDD international conference on Knowledge discovery and data mining. 2012.
'''

import gc
import warnings
from datetime import datetime
import multiprocessing as mp

import numpy as np
import pandas as pd
from numba import njit
from tqdm import tqdm

from utils import LoadSave, StreamDeque

# 设定全局随机种子，并且屏蔽warnings
GLOBAL_RANDOM_SEED = 2021
np.random.seed(GLOBAL_RANDOM_SEED)
warnings.filterwarnings('ignore')
###############################################################################

def compute_kpi_feats_single_step(data_pack, stream_deque):
    '''计算1个step的数据的统计特征，并更新deque参数'''
    tmp_feats = []

    # 解压数据值
    # *************
    kpi_id, timestep, sensor_val = data_pack
    tmp_feats.extend([timestep, int(kpi_id)])

    # 数据入队
    # *************
    stream_deque.push(timestep, sensor_val)

    # Stream统计特征抽取
    # *************
    # 计算滑窗均值
    for window_minutes in [5, 10, 16, 32]:
        tmp_feats.append(
            stream_deque.get_window_mean(int(window_minutes * 60))
        )

    # 计算滑窗标准差
    for window_minutes in [5, 10, 16, 32]:
        tmp_feats.append(
            stream_deque.get_window_std(int(window_minutes * 60))
        )

    # value shift
    for n_shift in [i for i in range(64)]:
        tmp_feats.append(
            (stream_deque.get_window_shift(n_shift) - sensor_val) / (1 + sensor_val)
        )

    # Moving Average


    # EWMA

    # Segment
    # *************
    

    # 更新deque参数
    # *************
    stream_deque.update()

    return tmp_feats


def compute_kpi_feats_df(df, stream_deque=None):
    '''训练数据特征构造接口'''

    # 基础元信息记录
    # ----------------
    df = df[['kpi_id', 'timestamp', 'value', 'label']]
    min_interval = int(np.min(np.diff(df['timestamp'].values)))

    target = df['label'].values
    df.drop(['label'], axis=1, inplace=True)

    # 数据入队 + 特征抽取
    # ----------------
    if stream_deque is None:
        stream_deque = StreamDeque(
            interval=min_interval,
            max_time_span=int(5 * 24 * 3600)  # 5 days
        )

    total_stat_feats = []
    for data_pack in df.values:
        total_stat_feats.append(
            compute_kpi_feats_single_step(data_pack, stream_deque)
        )

    # 特征组装
    # ----------------
    df_feats = np.vstack(total_stat_feats)
    df_feats_names = ['timestamp', 'kpi_id'] + \
        ['feat_{}'.format(i) for i in range(df_feats.shape[1] - 2)]
    df_feats = pd.DataFrame(df_feats, columns=df_feats_names)
    df_feats['label'] = target

    df_feats['kpi_id'] = df['kpi_id'].values
    df_feats['timestamp'] = df['timestamp'].values

    # 重排特征顺序
    # ----------------
    new_col_order = ['kpi_id'] + [item for item in df_feats.columns if item != 'kpi_id']
    df_feats = df_feats[new_col_order]

    return df_feats, stream_deque


if __name__ == '__main__':
    # 载入train_df数据
    # ----------------
    file_processor = LoadSave(dir_name='../cached_data/')
    train_df = file_processor.load_data(file_name='train_df.pkl')

    # 按kpi_id拆分训练数据
    # ----------------
    train_df_list = []
    unique_kpi_ids = train_df['kpi_id'].unique()
    for kpi_id in unique_kpi_ids:
        train_df_list.append(
            train_df[train_df['kpi_id'] == kpi_id].reset_index(drop=True)
        )

    del train_df
    gc.collect()

    # 测试特征工程
    # ----------------
    kpi_id = 0
    feat_df_tmp, stream_deque_tmp = compute_kpi_feats_df(
        train_df_list[kpi_id]
    )

    # 多进程并行特征工程
    # ----------------
    with mp.Pool(processes=mp.cpu_count(), maxtasksperchild=1) as p:
        meta_df_list = list(tqdm(p.imap(
            compute_kpi_feats_df, train_df_list
        ), total=len(train_df_list)))

    feats_df_list = []
    for i in range(len(train_df_list)):
        feats_df_list.append(meta_df_list[i][0])

    stream_deque_dict = {}
    for i in range(len(train_df_list)):
        stream_deque_dict[unique_kpi_ids[i]] = meta_df_list[i][1]

    # 存储特征数据
    # ----------------
    file_processor = LoadSave(dir_name='../cached_data/')
    file_processor.save_data(
        file_name='train_feats_list.pkl', data_file=feats_df_list
    )
    file_processor.save_data(
        file_name='train_stream_deque_dict.pkl', data_file=stream_deque_dict
    )

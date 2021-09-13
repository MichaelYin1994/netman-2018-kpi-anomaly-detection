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

def compute_feature_engineering_single_kpi(df):
    '''以Stream的形式给定KPI曲线的各类统计特征'''

    # 基础元信息记录
    # ----------------
    df = df[['kpi_id', 'timestamp', 'value', 'label']]
    min_interval = int(np.min(np.diff(df['timestamp'].values)))

    # 数据入队 + 特征抽取
    # ----------------
    stream_deque = StreamDeque(
        interval=min_interval,
        max_time_span=int(7 * 24 * 3600)  # 7 days
    )

    total_stat_feats = []
    for data_pack in df.values:
        tmp_feats = []

        # 解压数据值
        # *************
        kpi_id, timestep, sensor_val, label = data_pack
        tmp_feats.extend([kpi_id, timestep, label])

        # 数据入队
        # *************
        stream_deque.push(timestep, sensor_val)

        # 统计特征抽取
        # *************

        # 计算滑窗均值
        for window_minutes in [5, 10, 16, 32, 60, 120, 240, 360, 720, 1440]:
            tmp_feats.append(
                stream_deque.get_window_mean(int(window_minutes * 60))
            )

        # 计算滑窗标准差
        for window_minutes in [5, 10, 16, 32, 60, 120, 240, 360, 720, 1440]:
            tmp_feats.append(
                stream_deque.get_window_std(int(window_minutes * 60))
            )

        # value shift
        for n_shift in [i for i in range(256)]:
            tmp_feats.append(
                stream_deque.get_window_shift(n_shift)
            )

        # 特征保存
        # *************
        total_stat_feats.append(tmp_feats)

        # 更新deque
        # *************
        stream_deque.update()

    # 特征组装
    # ----------------
    df_feats_names = ['kpi_id', 'timestamp', 'label'] + \
        ['feat_{}'.format(i) for i in range(len(tmp_feats) - 3)]

    df_feats = np.vstack(total_stat_feats)
    df_feats = pd.DataFrame(df_feats, columns=df_feats_names)

    df_feats['kpi_id'] = df['kpi_id'].values
    df_feats['timestamp'] = df['timestamp'].values

    if 'label' in df.columns:
        df_feats['label'] = df['label'].values

    # 重排特征顺序
    new_col_order = ['kpi_id'] + [item for item in df_feats.columns if item != 'kpi_id']
    df_feats = df_feats[new_col_order]

    return df_feats


if __name__ == '__main__':
    # 载入train_df数据
    # ----------------
    file_processor = LoadSave(dir_name='../cached_data/')
    train_df = file_processor.load_data(file_name='train_df.pkl')

    # 按kpi_id拆分训练数据
    # ----------------
    unique_kpi_ids = train_df['kpi_id'].unique()

    train_df_list = []
    for kpi_id in unique_kpi_ids:
        train_df_list.append(
            train_df[train_df['kpi_id'] == kpi_id].reset_index(drop=True)
        )

    del train_df
    gc.collect()

    # 测试特征工程
    # ----------------
    kpi_id = 0
    feat_df_tmp = compute_feature_engineering_single_kpi(
        train_df_list[kpi_id]
    )

    # 多进程并行特征工程
    # ----------------
    with mp.Pool(processes=mp.cpu_count(), maxtasksperchild=1) as p:
        feats_df_list = list(tqdm(p.imap(
            compute_feature_engineering_single_kpi, train_df_list
        ), total=len(train_df_list)))

    # 存储特征数据
    # ----------------
    file_processor = LoadSave(dir_name='../cached_data/')
    file_processor.save_data(
        file_name='train_feats_list.pkl', data_file=feats_df_list
    )

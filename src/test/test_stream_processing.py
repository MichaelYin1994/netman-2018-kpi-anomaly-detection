#!/usr/bin/env python3
#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202109061216
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
本模块(test_stream_processing.py)用于测试流特征工程的正确性。
'''

import sys
sys.path.append('..')

import time
import warnings
import pandas as pd
from tqdm import tqdm
import numpy as np
import numba
from numba import njit
from numba.experimental import jitclass

from utils import StreamDeque

# 设定全局随机种子，并且屏蔽warnings
GLOBAL_RANDOM_SEED = 2021
np.random.seed(GLOBAL_RANDOM_SEED)
warnings.filterwarnings('ignore')

###############################################################################
def generate_simulation_data(n_points=10000, min_interval=20):
    '''生成kpi仿真数据'''
    sensor_vals = np.random.random(n_points)
    timestamp = np.array(
        [i * min_interval for i in range(4 * n_points)], dtype=np.int64
    )

    # 生成随机的时间戳
    rand_idx = np.arange(0, len(timestamp))
    np.random.shuffle(rand_idx)

    timestamp = timestamp[rand_idx[:n_points]]
    timestamp = np.sort(timestamp)

    # 组装数据为pandas DataFrame
    kpi_df = pd.DataFrame(
        {'timestamp': timestamp, 'value': sensor_vals,
         'kpi_id': np.ones((len(timestamp), ))}
    )

    return kpi_df


if __name__ == '__main__':
    # 生成kpi仿真数据（单条kpi曲线）
    # *******************
    N_POINTS = 3000000
    MIN_INTERVAL = 20
    MAX_TIME_SPAN = int(5 * 24 * 3600)
    WINDOW_SIZE = int(6 * 3600)
    df = generate_simulation_data(n_points=N_POINTS, min_interval=MIN_INTERVAL)
    df['value'] = (df['value'] - df['value'].mean()) / df['value'].std()

    # 流式抽取统计特征
    # *******************
    stream_deque = StreamDeque(
        interval=MIN_INTERVAL, max_time_span=MAX_TIME_SPAN
    )

    window_mean_results = []
    window_std_results = []
    window_shift_results = []
    window_count_results = []
    window_hog_1d_results = []
    for timestep, sensor_val in tqdm(df[['timestamp', 'value']].values):
        # 元素入队
        stream_deque.push(timestep, sensor_val)

        # 统计量计算
        window_mean_results.append(
            stream_deque.get_window_mean(WINDOW_SIZE)
        )

        window_std_results.append(
            stream_deque.get_window_std(WINDOW_SIZE)
        )

        # window_count_results.append(
        #     stream_deque.get_window_range_count(
        #         WINDOW_SIZE, 0.2, 0.5
        #     )
        # )

        # window_hog_1d_results.append(
        #     stream_deque.get_window_hog_1d(
        #         WINDOW_SIZE, -60, 60, 16
        #     )
        # )
        # 空间拓展
        stream_deque.update()

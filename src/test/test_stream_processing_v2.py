#!/usr/bin/env python3
#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202109110003
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
本模块(test_stream_processing_v2.py)用于测试流特征工程的正确性。
'''


import sys
sys.path.append('..')

import warnings
import pandas as pd
from tqdm import tqdm
import numpy as np
from numba import njit
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# 设定全局随机种子，并且屏蔽warnings
GLOBAL_RANDOM_SEED = 2021
np.random.seed(GLOBAL_RANDOM_SEED)
warnings.filterwarnings('ignore')

def generate_simulation_data(n_points=10000, min_interval=20):
    '''生成KPI仿真数据'''
    sensor_vals = np.linspace(0, np.pi * 10.75, n_points, endpoint=False)
    sensor_vals = np.cos(sensor_vals) + np.sin(sensor_vals * 5) * 0.2
    sensor_vals += np.cos(sensor_vals * 2) * 2.2
    sensor_vals += np.random.uniform(-2, 1, n_points)

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


@njit
def shrink_window_time_span(timestamp, start, end, max_time_span):
    '''收缩串口窗口到指定允许时间窗口范围'''
    time_gap = timestamp[end] - timestamp[start]

    if time_gap > max_time_span:
        while(start <= end and time_gap > max_time_span):
            start += 1
            time_gap = timestamp[end] - timestamp[start]

    return start, end


@njit
def update_window_mean_params(sensor_vals, old_start, old_end,
                              new_start, new_end, window_sum):
    if np.isnan(window_sum):
        window_sum = np.sum(sensor_vals[old_start:(old_end + 1)])
    else:
        window_sum += sensor_vals[new_end]

    # 统计量修正
    window_sum_delta = 0

    for i in range(old_start, new_start):
        window_sum_delta -= sensor_vals[i]

    # 统计量更新
    dist2end = new_end - new_start + 1
    window_sum = window_sum + window_sum_delta
    mean_res = window_sum / dist2end

    return dist2end, window_sum, mean_res


@njit
def update_window_std_params(sensor_vals, old_start, old_end,
                             new_start, new_end, window_sum, window_squre_sum):
    if np.isnan(window_sum):
        window_sum = np.sum(sensor_vals[old_start:(old_end + 1)])
        window_squre_sum = np.sum(sensor_vals[old_start:(old_end + 1)]**2)
    else:
        window_sum += sensor_vals[new_end]
        window_squre_sum += sensor_vals[new_end]**2

    # 统计量修正
    window_sum_delta = 0
    window_squre_sum_delta = 0

    for i in range(old_start, new_start):
        window_sum_delta -= sensor_vals[i]
        window_squre_sum_delta -= sensor_vals[i]**2

    # 统计量更新
    dist2end = new_end - new_start + 1
    window_sum = window_sum + window_sum_delta
    window_squre_sum = window_squre_sum + window_squre_sum_delta
    std_res = np.sqrt(
        window_squre_sum / dist2end - (window_sum / dist2end)**2
    )

    return dist2end, window_sum, window_squre_sum, std_res


@njit
def update_window_range_count_params(sensor_vals,
                                     old_start, old_end,
                                     new_start, new_end,
                                     low, high, low_count, high_count):

    if sensor_vals[new_end] > high:
         high_count += 1
    elif sensor_vals[new_end] < low:
        low_count += 1

    # 时间窗口放缩，统计量修正
    window_low_count_delta = 0
    window_high_count_delta = 0

    for i in range(old_start, new_start):
        if sensor_vals[i] > high:
            window_high_count_delta -= 1
        elif sensor_vals[i] < low:
            window_low_count_delta -= 1

    # 统计量更新
    dist2end = new_end - new_start + 1
    low_count = low_count + window_low_count_delta
    high_count = high_count + window_high_count_delta

    return dist2end, low_count, high_count


class StreamDeque():
    '''对于时序流数据（stream data）的高效存储与基础特征抽取方法的实现。

    采用numpy array模拟deque，deque保存指定时间区间范围内的时序值。每当时间
    范围不满足条件的时候，通过指针移动模拟队尾元素出队；当array满的时候，
    动态重新分配deque的内存空间。

    我们实现的ArrayDeque在设计时便考虑了数据流时间戳不均匀的问题，通过移动指针
    的方式高效的实现元素的入队和出队，保证deque内只有指定时间范围内的时序数据。

    ArrayDeque的实现中，同时设计了多种针对流数据的特征抽取算法。包括：
    - 给定时间窗口内均值(mean)抽取算法。
    - 给定时间窗口内标准差(std)抽取算法。
    - 给定时间窗口内的range count分布抽取。
    - 给定时间窗口内的HOG-1D特征[2]抽取。
    - 时间shift特征。
    - EWMA，Holt预测方法。

    时序stream特征抽取时，若是时间戳连续，则算法可以做到O(1)时间与空间复杂度，
    但是由于实际场景中日志采集的时间戳不一定是连续的，因此抽取指定窗口内的统计量
    需要先进行窗口放缩的操作，因此时间复杂度不再是O(1)，但是我们的实现仍然保证了
    线性时间复杂度的统计量抽取。

    @Attributes:
    ----------
    interval: {int-like}
        元素与元素之间最小的时间间隔单位，默认为秒。
    max_time_span: {int-like}
        deque首元素与尾元素最大允许时间差，默认单位为秒。
    deque_timestamp: {array-like}
        用于存储unix时间戳的数组，模拟双端队列。
    deque_vals: {array-like}
        用于存储实际传感器读数的数组，模拟双端队列。
    deque_size: {int-like}
        deque的大小。
    deque_front: {int-like}
        用于模拟deque范围的deque的头指针。
    deque_rear: {int-like}
        用于模拟deque范围的deque的尾指针，永远指向deque最后一个有值元素索引的
        下一个元素索引。
    deque_stats: {dict-like}
        用于保持stream计算统计量时的一些基础统计信息。

    @References:
    ----------
    [1] https://www.kaggle.com/lucasmorin/running-algos-fe-for-fast-inference
    [2] Zhao, Jiaping, and Laurent Itti. "Classifying time series using local descriptors with hybrid sampling." IEEE Transactions on Knowledge and Data Engineering 28.3 (2015): 623-637.
    [3] Rakthanmanon, Thanawin, et al. "Searching and mining trillions of time series subsequences under dynamic time warping." Proceedings of the 18th ACM SIGKDD international conference on Knowledge discovery and data mining. 2012.
    [4] Zhao, Nengwen, et al. "Label-less: A semi-automatic labelling tool for kpi anomalies." IEEE INFOCOM 2019-IEEE Conference on Computer Communications. IEEE, 2019.
    '''
    def __init__(self, interval=20, max_time_span=3600):
        self.interval = interval
        self.max_time_span = max_time_span
        self.deque_size = int(max_time_span // interval * 2) + 1
        self.deque_stats =  {}

        self.deque_timestamp = np.zeros((self.deque_size, ), dtype=np.int64)
        self.deque_vals = np.zeros((self.deque_size, ), dtype=np.float64)
        self.deque_front, self.deque_rear = 0, 0

    def __len__(self):
        return self.deque_rear - self.deque_front

    def push(self, timestep, x):
        '''将一组元素入队，并且依据deque尾部与首部的元素时间戳，调整deque指针'''
        # (timestamp, x) 入队
        self.deque_timestamp[self.deque_rear] = timestep
        self.deque_vals[self.deque_rear] = x

        # 不满足deque时间窗需求，收缩deque范围
        new_start, _ = shrink_window_time_span(
            self.deque_timestamp,
            self.deque_front,
            self.deque_rear,
            self.max_time_span
        )

        self.deque_front = new_start
        self.deque_rear += 1

    def update(self):
        '''若deque满，则动态调整队列数组内存空间'''
        if self.is_full():
            rear = len(self.deque_timestamp[self.deque_front:])

            # 原地调整空间，防止内存泄漏
            self.deque_timestamp[:rear] = self.deque_timestamp[self.deque_front:]
            self.deque_timestamp[rear:] = 0

            self.deque_vals[:rear] = self.deque_vals[self.deque_front:]
            self.deque_vals[rear:] = 0.0

            # 更新头尾指针
            self.deque_front, self.deque_rear = 0, rear

    def is_full(self):
        '''判断deque是否满'''
        return self.deque_rear >= self.deque_size

    def check_window_size(self, window_size):
        '''检查window_size参数是否合法'''
        if window_size > self.max_time_span or window_size < self.interval:
            raise ValueError('Invalid input window size !')

    def get_window_mean(self, window_size):
        '''抽取window_size范围内的mean统计量'''
        self.check_window_size(window_size)

        # 载入stream参数（加法hash计算索引）
        field_name = hash(window_size) + 0

        if field_name in self.deque_stats:
            dist2end, window_sum = self.deque_stats[field_name]
        else:
            window_sum = np.nan
            dist2end = self.deque_rear - self.deque_front - 1

        start = int(self.deque_rear - dist2end - 1)
        end = int(self.deque_rear - 1)

        # 重新计算窗口参数
        new_start, new_end = shrink_window_time_span(
            self.deque_timestamp, start, end, self.max_time_span
        )

        # 更新窗口统计量
        dist2end, window_sum, mean_res = update_window_mean_params(
            self.deque_vals, start, end, new_start, new_end, window_sum
        )

        # 更新预置参数
        new_params = [dist2end, window_sum]
        self.deque_stats[field_name] = new_params

        return mean_res

    def get_window_std(self, window_size):
        '''抽取window_size范围内的mean统计量'''
        self.check_window_size(window_size)

        # 载入stream参数（加法hash计算索引）
        field_name = hash(window_size) + 1

        if field_name in self.deque_stats:
            dist2end, window_sum, window_squre_sum = self.deque_stats[field_name]
        else:
            window_sum, window_squre_sum = np.nan, np.nan
            dist2end = self.deque_rear - self.deque_front - 1

        start = int(self.deque_rear - dist2end - 1)
        end = int(self.deque_rear - 1)

        # 重新计算窗口参数
        new_start, new_end = shrink_window_time_span(
            self.deque_timestamp, start, end, self.max_time_span
        )

        # 更新窗口统计量
        dist2end, window_sum, window_squre_sum, std_res = update_window_std_params(
            self.deque_vals, start, end, new_start, new_end, window_sum, window_squre_sum
        )

        # 更新预置参数
        new_params = [dist2end, window_sum, window_squre_sum]
        self.deque_stats[field_name] = new_params

        return std_res

    def get_window_range_count_ratio(self, window_size, low, high):
        '''抽取window_size内的位于low与high闭区间内部数据的比例'''
        # 输入检查
        if low > high:
            raise ValueError('Invalid value range !')
        self.check_window_size(window_size)

        # 载入stream参数（加法hash计算索引）
        field_name = hash(window_size) + hash(low) + hash(high)

        if field_name in self.deque_stats:
            dist2end, low_count, high_count = self.deque_stats[field_name]
        else:
            low_count, high_count = 0, 0
            dist2end = self.deque_rear - self.deque_front - 1

        start = int(self.deque_rear - dist2end - 1)
        end = int(self.deque_rear - 1)

        # 重新计算窗口参数
        new_start, new_end = shrink_window_time_span(
            self.deque_timestamp, start, end, self.max_time_span
        )

        # 重新计算参数
        dist2end, low_count, high_count = update_window_range_count_params(
            self.deque_vals,
            start, end,
            new_start, new_end,
            low, high,
            low_count, high_count
        )

        # 更新预置参数
        new_params = [dist2end, low_count, high_count]
        self.deque_stats[field_name] = new_params
        count_precent = (dist2end - low_count - high_count) / dist2end

        return count_precent

    def get_n_shift(self, n_shift):
        '''抽取当前时刻给定上n_shift个timestep的数据的值'''
        if n_shift < 0:
            raise ValueError('Invalid n_shift parameter !')

        if n_shift > (self.deque_rear - self.deque_front):
            return np.nan
        else:
            return self.deque_vals[self.deque_rear - n_shift - 1]

    def get_window_timestamp_values(self, window_size):
        '''抽取指定window_size内的(timestamp, value)数组'''
        self.check_window_size(window_size)

        # 载入stream参数（加法hash计算索引）
        field_name = hash(window_size) - 65536

        if field_name in self.deque_stats:
            dist2end = self.deque_stats[field_name]
        else:
            dist2end = self.deque_rear - self.deque_front - 1

        start = int(self.deque_rear - dist2end - 1)
        end = int(self.deque_rear - 1)

        # 重新计算窗口参数
        new_start, new_end = shrink_window_time_span(
            self.deque_timestamp, start, end, window_size
        )
        dist2end = new_end - new_start + 1

        # 更新预置参数
        self.deque_stats[field_name] = dist2end

        window_timestamp = self.deque_timestamp[new_start:self.deque_rear]
        window_sensor_vals = self.deque_vals[new_start:self.deque_rear]

        return window_timestamp, window_sensor_vals

    def get_prediction_exponential_weighted_mean(self, alpha=0.25, alpha_type=0):
        '''以O(1)时间复杂度计算Exponential Weighted Average Prediction结果'''
        # aplha_type:
        # 0: ordinary alpha
        # 1: span
        # 2: Center of life
        # 3: Half-life
        if alpha_type == 1:
            if alpha < 1:
                raise ValueError('Invalid span parameter '
                                 '(\alpha should >= 1)!')
            alpha = 2 / (alpha + 2)
        elif alpha_type == 2:
            if alpha < 0:
                raise ValueError('Invalid center of mass'
                                 ' parameter (\alpha should >= 0)!')
                alpha = 1 / (1 + alpha)
        elif alpha_type == 3:
            if alpha < 0:
                raise ValueError('Invalid half of life'
                                 ' parameter (\alpha should >= 0)!')
            alpha = 1 - np.exp(np.log(0.5) / alpha)

        # 载入stream参数（加法hash计算索引）
        field_name = hash(alpha) + hash(alpha_type)

        if field_name in self.deque_stats:
            l_t_1 = self.deque_stats[field_name]
        else:
            l_t_1 = 0

        if len(self) == 1:
            y_t_1 = 1 / alpha * self.deque_vals[self.deque_front]
        else:
            y_t_1 = self.get_n_shift(1)

        # prediction
        l_t = alpha * y_t_1 + (1 - alpha) * l_t_1
        y_hat_t = l_t

        # 更新预置参数
        self.deque_stats[field_name] = y_hat_t

        return y_hat_t

    def get_prediction_holt(self, alpha, beta):
        '''以O(1)时间复杂度计算Holt Prediction结果'''
        # 载入stream参数（加法hash计算索引）
        field_name = hash(alpha + beta) + 256

        if field_name in self.deque_stats:
            l_t_1, b_t_1 = self.deque_stats[field_name]
        else:
            l_t_1, b_t_1 = 0, 0

        if len(self) == 1:
            y_t_1 = 1 / alpha * self.deque_vals[self.deque_front]
        else:
            y_t_1 = self.get_n_shift(1)

        # prediction
        l_t = alpha * y_t_1 + (1 - alpha) * (l_t_1 + b_t_1)
        b_t = beta * (l_t - l_t_1) + (1 - beta) * b_t_1
        y_hat_t = l_t + b_t

        # 更新预置参数
        self.deque_stats[field_name] = [l_t, b_t]

        return y_hat_t


if __name__ == '__main__':
    # 生成kpi仿真数据（单条kpi曲线）
    # *******************
    IS_VISUALIZING = False
    N_POINTS = 300000
    MIN_INTERVAL = 30
    MAX_TIME_SPAN = int(6 * 60 * 60)
    WINDOW_SIZE = 1200
    df = generate_simulation_data(
        n_points=N_POINTS, min_interval=MIN_INTERVAL
    )

    # Analysis
    # *******************
    if IS_VISUALIZING:
        plt.close('all')
        fig, ax = plt.subplots(figsize=(12, 4))

        ax.grid(True)
        ax.set_xlim(0, len(df))
        ax.set_xlabel('Timestamp', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.set_title('Real-time KPI', fontsize=10)
        ax.tick_params(axis="both", labelsize=10)

        ax.plot(df['value'].values, linestyle='--', markersize=2.5,
                color='k', marker='.', label="KPI curve")

        ax.legend(fontsize=10)
        plt.tight_layout()

    # 流式抽取统计特征
    # *******************
    stream_deque = StreamDeque(
        interval=MIN_INTERVAL, max_time_span=MAX_TIME_SPAN
    )

    window_feats = []
    for timestep, sensor_val in tqdm(df[['timestamp', 'value']].values):
        # 元素入队
        # --------------
        stream_deque.push(timestep, sensor_val)

        # 统计量计算
        # --------------
        # window_feats.append(
        #     stream_deque.get_window_mean(WINDOW_SIZE)
        # )

        # window_feats.append(
        #     stream_deque.get_window_std(WINDOW_SIZE)
        # )

        window_feats.append(
            stream_deque.get_window_range_count_ratio(WINDOW_SIZE, -0.1, 0.1)
        )

        # timestamp, seneor_vals = stream_deque.get_window_timestamp_values(WINDOW_SIZE)

        # window_feats.append(
        #     stream_deque.get_prediction_exponential_weighted_mean(0.7)
        # )

        # window_feats.append(
        #     stream_deque.get_prediction_exponential_weighted_mean(0.7)
        # )


        # window_feats.append(
        #     stream_deque.get_prediction_holt(0.75, 0.8)
        # )

        # 空间拓展
        # --------------
        stream_deque.update()


    # plt.close('all')
    # fig, ax = plt.subplots(figsize=(12, 4))

    # ax.grid(True)
    # ax.set_xlabel('Timestamp', fontsize=10)
    # ax.set_ylabel('Value', fontsize=10)
    # ax.set_title('Real-time KPI', fontsize=10)
    # ax.tick_params(axis="both", labelsize=10)

    # ax.plot(df['value'].values[:500], linestyle='--', markersize=2.5,
    #         color='k', marker='.', label="KPI curve")
    # ax.plot(window_feats[:500], linestyle='--', markersize=2.5,
    #         color='r', marker='.', label="KPI curve")

    # ax.legend(fontsize=10)
    # plt.tight_layout()

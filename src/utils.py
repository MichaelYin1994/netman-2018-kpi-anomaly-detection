#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202108311136
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
数据处理与特征工程辅助代码。
'''

import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import numba
from numba.experimental import jitclass
from numba import njit
from tqdm import tqdm

from sklearn.metrics import auc, precision_recall_curve

warnings.filterwarnings('ignore')
###############################################################################
@njit
def adjust_predict_label(true_label, pred_label, delay):
    '''按照delay参数与实际标签，调整预测标签'''
    split_idx = np.arange(0, len(pred_label) - 1) + 1
    split_cond = (true_label[1:] != true_label[:-1])
    split_idx = split_idx[split_cond]

    # 扫描数组，按照要求重组样本标签
    # http://iops.ai/competition_detail/?competition_id=5&flag=1
    front, is_segment_anomaly = 0, true_label[0] == 1
    adjusted_pred_label = pred_label.copy()
    for rear in split_idx:
        if is_segment_anomaly:
            if 1 in pred_label[front:min(front + delay + 1, rear)]:
                adjusted_pred_label[front:rear] = 1
            else:
                adjusted_pred_label[front:rear] = 0

        is_segment_anomaly = not is_segment_anomaly
        front = rear

    if is_segment_anomaly:
        if 1 in pred_label[front:]:
            adjusted_pred_label[front:rear] = 1
        else:
            adjusted_pred_label[front:rear] = 0

    return adjusted_pred_label


@njit
def reconstruct_label(timestamp, label):
    '''依据最小采样间隔，重新组织预测标签'''
    timestamp_sorted_idx = np.argsort(timestamp)
    timestamp_sorted = timestamp[timestamp_sorted_idx]
    label_sorted = label[timestamp_sorted_idx]

    # 获取样本最小采样间隔
    min_interval = np.min(np.diff(timestamp_sorted))

    # 依据最小采样间隔，重构标签与timestamp数组
    new_timestamp = np.arange(
        int(np.min(timestamp)),
        int(np.max(timestamp)) + int(min_interval),
        int(min_interval)
    )
    new_label = np.zeros((len(new_timestamp), ))
    new_label_idx = (timestamp_sorted - timestamp_sorted[0]) // min_interval
    new_label[new_label_idx] = label_sorted

    return label


@njit
def njit_f1(y_true_label, y_pred_label):
    '''计算F1分数，使用njit加速计算'''
    # https://www.itread01.com/content/1544007604.html
    tp = np.sum(np.logical_and(np.equal(y_true_label, 1),
                               np.equal(y_pred_label, 1)))
    fp = np.sum(np.logical_and(np.equal(y_true_label, 0),
                               np.equal(y_pred_label, 1)))
    # tn = np.sum(np.logical_and(np.equal(y_true, 1),
    #                            np.equal(y_pred_label, 0)))
    fn = np.sum(np.logical_and(np.equal(y_true_label, 1),
                               np.equal(y_pred_label, 0)))

    if (tp + fp) == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if (tp + fn) == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    if (precision + recall) == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def pr_auc_score(y_true, y_pred):
    '''PR Curve AUC计算方法'''
    precision, recall, _ =  precision_recall_curve(
        y_true.reshape(-1, 1), y_pred.reshape(-1, 1)
    )
    auc_score = auc(recall, precision)

    return auc_score


def evaluate_df_score(true_df, pred_df, delay=7):
    '''依据比赛[1]与论文[2]的评测方法计算KPI预测结果的分数。DataFrame必须
    包括3列："kpi_id", "label"与"timestamp"。其中timestamp列为unix-like时间戳。

    @Parameters:
    ----------
    true_df: {pandas DataFrame}
        标签DataFrame，包含"kpi_id", "label"与"timestamp"三列。
    pred_df: {pandas DataFrame}
        预测结果的DataFrame，包含"kpi_id", "label"与"timestamp"三列。
    delay: {int}
        容错范围，delay \in [0, +\inf]，delay越小评测越严格。

    @Returens:
    ----------
    dict类型，不同kpi_id计算的分数与整体计算的分数。

    @References:
    ----------
    [1] http://iops.ai/competition_detail/?competition_id=5&flag=1
    [2] Zhao, Nengwen, et al. "Label-less: A semi-automatic labelling tool for kpi anomalies." IEEE INFOCOM 2019-IEEE Conference on Computer Communications. IEEE, 2019.

    TODO(zhuoyin94@163.com):
    ----------
    官方评测Metric在未发生故障的segment部分存在不合理设计。更合理
    的Metric应该是按整segment为单位计算F1。
    '''
    if len(true_df) != len(pred_df):
        raise ValueError('Predicting result shape mismatch !')
    for name in ['kpi_id', 'label', 'timestamp']:
        if name not in true_df.columns or name not in pred_df.columns:
            raise ValueError('{} not present in DataFrame !'.format(name))

    # 计算每条KPI曲线的分数
    unique_kpi_ids_list = true_df['kpi_id'].unique().tolist()
    adjusted_pred_label_list, true_label_list = [], []
    f1_score, precision_score, recall_score = [], [], []

    for kpi_id in unique_kpi_ids_list:
        true_df_tmp = true_df[true_df['kpi_id'] == kpi_id]
        pred_df_tmp = pred_df[pred_df['kpi_id'] == kpi_id]

        true_label_tmp = reconstruct_label(
            true_df_tmp['timestamp'].values,
            true_df_tmp['label'].values,
        )
        pred_label_tmp = reconstruct_label(
            pred_df_tmp['timestamp'].values,
            pred_df_tmp['label'].values,
        )
        pred_label_adjusted_tmp = adjust_predict_label(
            true_label_tmp, pred_label_tmp, delay
        )
        f1, precision, recall = njit_f1(
            true_label_tmp, pred_label_adjusted_tmp
        )

        # 保留计算结果
        true_label_list.append(true_label_tmp)
        adjusted_pred_label_list.append(pred_label_adjusted_tmp)

        f1_score.append(f1)
        precision_score.append(precision)
        recall_score.append(recall)

    # 计算整体的分数（体现算法通用性）
    true_label_array = np.concatenate(true_label_list)
    adjusted_pred_label_array = np.concatenate(adjusted_pred_label_list)

    f1_total, precision_total, recall_total = njit_f1(
        true_label_array, adjusted_pred_label_array
    )

    # 保存并返回全部计算结果
    score_dict = {}
    score_dict['f1_score_list'] = f1_score
    score_dict['precision_score_list'] = precision_score
    score_dict['recall_score_list'] = recall_score
    score_dict['total_score'] = [f1_total, precision_total, recall_total]

    return score_dict


class LoadSave():
    '''以*.pkl格式，利用pickle包存储各种形式（*.npz, list etc.）的数据。

    @Attributes:
    ----------
        dir_name: {str-like}
            数据希望读取/存储的路径信息。
        file_name: {str-like}
            希望读取与存储的数据文件名。
        verbose: {int-like}
            是否打印存储路径信息。
    '''
    def __init__(self, dir_name=None, file_name=None, verbose=1):
        if dir_name is None:
            self.dir_name = './data_tmp/'
        else:
            self.dir_name = dir_name
        self.file_name = file_name
        self.verbose = verbose

    def save_data(self, dir_name=None, file_name=None, data_file=None):
        '''将data_file保存到dir_name下以file_name命名。'''
        if data_file is None:
            raise ValueError('LoadSave: Empty data_file !')

        if dir_name is None or not isinstance(dir_name, str):
            dir_name = self.dir_name
        if file_name is None:
            file_name = self.file_name
        if not isinstance(file_name, str) or not file_name.endswith('.pkl'):
            raise ValueError('LoadSave: Invalid file_name !')

        # 保存数据以指定名称到指定路径
        full_name = dir_name + file_name
        with open(full_name, 'wb') as file_obj:
            pickle.dump(data_file, file_obj, protocol=4)

        if self.verbose:
            print('[INFO] {} LoadSave: Save to dir {} with name {}'.format(
                str(datetime.now())[:-4], dir_name, file_name))

    def load_data(self, dir_name=None, file_name=None):
        '''从指定的dir_name载入名字为file_name的文件到内存里。'''
        if dir_name is None or not isinstance(dir_name, str):
            dir_name = self.dir_name
        if file_name is None:
            file_name = self.file_name
        if not isinstance(file_name, str) or not file_name.endswith('.pkl'):
            raise ValueError('LoadSave: Invalid file_name !')

        # 从指定路径导入指定文件名的数据
        full_name = dir_name + file_name
        with open(full_name, 'rb') as file_obj:
            data_loaded = pickle.load(file_obj)

        if self.verbose:
            print('[INFO] {} LoadSave: Load from dir {} with name {}'.format(
                str(datetime.now())[:-4], dir_name, file_name))
        return data_loaded


def basic_feature_report(data_table, quantile=None):
    '''抽取Pandas的DataFrame的基础信息。'''
    if quantile is None:
        quantile = [0.25, 0.5, 0.75, 0.95, 0.99]

    # 基础统计数据
    data_table_report = data_table.isnull().sum()
    data_table_report = pd.DataFrame(data_table_report, columns=['#missing'])

    data_table_report['#uniques'] = data_table.nunique(dropna=False).values
    data_table_report['types'] = data_table.dtypes.values
    data_table_report.reset_index(inplace=True)
    data_table_report.rename(columns={'index': 'feature_name'}, inplace=True)

    # 分位数统计特征
    data_table_description = data_table.describe(quantile).transpose()
    data_table_description.reset_index(inplace=True)
    data_table_description.rename(
        columns={'index': 'feature_name'}, inplace=True)
    data_table_report = pd.merge(
        data_table_report, data_table_description,
        on='feature_name', how='left')

    return data_table_report


class LiteModel:
    '''将模型转换为Tensorflow Lite模型，提升推理速度。目前仅支持Keras模型转换。

    @Attributes:
    ----------
    interpreter: {Tensorflow lite transformed object}
        利用tf.lite.interpreter转换后的Keras模型。

    @References:
    ----------
    [1] https://medium.com/@micwurm/using-tensorflow-lite-to-speed-up-predictions-a3954886eb98
    '''

    @classmethod
    def from_file(cls, model_path):
        '''类方法。用于model_path下的模型，一般为*.h5模型。'''
        return LiteModel(tf.lite.Interpreter(model_path=model_path))

    @classmethod
    def from_keras_model(cls, kmodel):
        '''类方法。用于直接转换keras模型。不用实例化类可直接调用该方法，返回
        被转换为tf.lite形式的Keras模型。

        @Attributes:
        ----------
        kmodel: {tf.keras model}
            待转换的Keras模型。

        @Returens:
        ----------
        经过转换的Keras模型。
        '''
        converter = tf.lite.TFLiteConverter.from_keras_model(kmodel)
        tflite_model = converter.convert()
        return LiteModel(tf.lite.Interpreter(model_content=tflite_model))

    def __init__(self, interpreter):
        '''为经过tf.lite.interpreter转换的模型构建构造输入输出的关键参数。

        TODO(zhuoyin94@163.com):
        ----------
        [1] 可添加关键字，指定converter选择采用INT8量化还是混合精度量化。
        [2] 可添加关键字，指定converter选择量化的方式：低延迟还是高推理速度？
        '''
        self.interpreter = interpreter
        self.interpreter.allocate_tensors()

        input_det = self.interpreter.get_input_details()[0]
        output_det = self.interpreter.get_output_details()[0]
        self.input_index = input_det['index']
        self.output_index = output_det['index']
        self.input_shape = input_det['shape']
        self.output_shape = output_det['shape']
        self.input_dtype = input_det['dtype']
        self.output_dtype = output_det['dtype']

    def predict(self, inp):
        inp = inp.astype(self.input_dtype)
        count = inp.shape[0]
        out = np.zeros((count, self.output_shape[1]), dtype=self.output_dtype)
        for i in range(count):
            self.interpreter.set_tensor(self.input_index, inp[i:i+1])
            self.interpreter.invoke()
            out[i] = self.interpreter.get_tensor(self.output_index)[0]
        return out

    def predict_single(self, inp):
        ''' Like predict(), but only for a single record. The input data can be a Python list. '''
        inp = np.array([inp], dtype=self.input_dtype)
        self.interpreter.set_tensor(self.input_index, inp)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_index)
        return out[0]


@njit
def update_window_degree_bin_count_params(timestamp, sensor_vals,
                                          start, end, low, high,
                                          bin_count, interval, max_time_span):
    # 生成histogram窗口边界
    bin_range = np.linspace(low, high, len(bin_count) + 1)

    # 计算当前样本的Gradient
    y_delta = (sensor_vals[end] - sensor_vals[max(end-1, 0)])
    x_delta = (timestamp[end] - timestamp[max(end-1, 0)]) / interval
    if x_delta == 0:
        degree= 0
    else:
        degree = np.rad2deg(np.arctan(y_delta / x_delta))

    # 将Gradient映射到bin上去
    for i in range(1, len(bin_count) + 1):
        if degree > bin_range[i-1] and degree <= bin_range[i]:
            bin_count[i-1] += 1
            break

    # 时间窗口放缩，统计量修正
    time_gap = timestamp[end] - timestamp[start]

    if time_gap > max_time_span:
        while(start <= end and time_gap > max_time_span):

            # 统计量更新
            y_delta = (sensor_vals[min(end, start+1)] - sensor_vals[start])
            x_delta = (timestamp[min(end, start+1)] - timestamp[start]) / interval
            if x_delta == 0:
                degree= 0
            else:
                degree = np.rad2deg(np.arctan(y_delta / x_delta))

            for i in range(1, len(bin_count) + 1):
                if degree > bin_range[i-1] and degree <= bin_range[i]:
                    bin_count[i-1] -= 1
                    break

            start += 1
            time_gap = timestamp[end] - timestamp[start]
    dist2end = end - start + 1

    return dist2end, bin_count


@njit
def update_window_range_count_params(timestamp, sensor_vals,
                                     start, end, low, high,
                                     low_count, high_count, max_time_span):
    # 时间窗口放缩，统计量修正
    window_low_count_delta = 0
    window_high_count_delta = 0
    time_gap = timestamp[end] - timestamp[start]

    if sensor_vals[end] > high:
        high_count += 1
    elif sensor_vals[end] < low:
        low_count += 1

    if time_gap > max_time_span:
        while(start <= end and time_gap > max_time_span):
            if sensor_vals[start] > high:
                window_high_count_delta -= 1
            elif sensor_vals[start] < low:
                window_low_count_delta -= 1
            start += 1
            time_gap = timestamp[end] - timestamp[start]

    # 统计量更新
    dist2end = end - start + 1
    low_count = low_count + window_low_count_delta
    high_count = high_count + window_high_count_delta

    return dist2end, low_count, high_count


@njit
def update_window_std_params(timestamp, sensor_vals,
                             start, end, window_sum, window_squre_sum,
                             max_time_span):
    if np.isnan(window_sum):
        window_sum = np.sum(sensor_vals[start:(end + 1)])
        window_squre_sum = np.sum(sensor_vals[start:(end + 1)]**2)
    else:
        window_sum += sensor_vals[end]
        window_squre_sum += sensor_vals[end]**2

    # 时间窗口放缩，统计量修正
    window_sum_delta = 0
    window_squre_sum_delta = 0
    time_gap = timestamp[end] - timestamp[start]

    if time_gap > max_time_span:
        while(start <= end and time_gap > max_time_span):
            window_sum_delta -= sensor_vals[start]
            window_squre_sum_delta -= sensor_vals[start]**2
            start += 1
            time_gap = timestamp[end] - timestamp[start]

    # 统计量更新
    dist2end = end - start + 1
    window_sum = window_sum + window_sum_delta
    window_squre_sum = window_squre_sum + window_squre_sum_delta
    std_res = np.sqrt(
        window_squre_sum / dist2end - (window_sum / dist2end)**2
    )

    return dist2end, window_sum, window_squre_sum, std_res


@njit
def update_window_mean_params(timestamp, sensor_vals,
                              start, end, window_sum,
                              max_time_span):
    if np.isnan(window_sum):
        window_sum = np.sum(sensor_vals[start:(end + 1)])
    else:
        window_sum += sensor_vals[end]

    # 时间窗口放缩，统计量修正
    window_sum_delta = 0
    time_gap = timestamp[end] - timestamp[start]

    if time_gap > max_time_span:
        while(start <= end and time_gap > max_time_span):
            window_sum_delta -= sensor_vals[start]
            start += 1
            time_gap = timestamp[end] - timestamp[start]

    # 统计量更新
    dist2end = end - start + 1
    window_sum = window_sum + window_sum_delta
    mean_res = window_sum / dist2end

    return dist2end, window_sum, mean_res


@njit
def update_window_params(timestamp, sensor_vals,
                         start, end, max_time_span):
    # 时间窗口放缩，统计量修正
    time_gap = timestamp[end] - timestamp[start]

    if time_gap > max_time_span:
        while(start <= end and time_gap > max_time_span):
            start += 1
            time_gap = timestamp[end] - timestamp[start]

    # 统计量更新
    dist2end = end - start + 1

    return dist2end


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
        front, rear = self.deque_front, self.deque_rear

        time_gap = self.deque_timestamp[rear] - self.deque_timestamp[front]
        if time_gap > self.max_time_span:
            while(front <= rear and time_gap > self.max_time_span):
                front += 1
                time_gap = self.deque_timestamp[rear] - self.deque_timestamp[front]

        self.deque_front = front
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

    def get_values(self):
        return self.deque_vals[self.deque_front:self.deque_rear]

    def get_timestamp(self):
        return self.deque_timestamp[self.deque_front:self.deque_rear]

    def get_window_timestamp_values(self, window_size):
        '''获取指定window_size内的(timestamp, value)数组'''
        self.check_window_size(window_size)

        # 载入stream参数（加法hash计算索引）
        field_name = hash(window_size) - 1

        if field_name in self.deque_stats:
            dist2end = self.deque_stats[field_name]
        else:
            dist2end = self.deque_rear - self.deque_front - 1

        start = int(self.deque_rear - dist2end - 1)
        end = int(self.deque_rear - 1)

        # 更新窗口参数
        dist2end = update_window_params(
            self.deque_timestamp,
            self.deque_vals,
            start, end, window_size
        )

        # 更新预置参数
        new_params = np.array(
            [dist2end], dtype=np.int64
        )

        return end - dist2end + 1

    def get_window_mean(self, window_size=120):
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

        # 重新计算参数
        dist2end, window_sum, mean_res = update_window_mean_params(
            self.deque_timestamp,
            self.deque_vals,
            start, end, window_sum, window_size
        )

        # 更新预置参数
        new_params = np.array(
            [dist2end, window_sum], dtype=np.float64
        )
        self.deque_stats[field_name] = new_params

        return mean_res

    def get_window_std(self, window_size=120):
        '''抽取window_size范围内std统计量'''
        self.check_window_size(window_size)

        # 载入stream参数（加法hash计算索引）
        field_name = hash(window_size) + 1

        if field_name in self.deque_stats:
            dist2end, window_sum, window_squre_sum = self.deque_stats[field_name]
        else:
            window_sum = np.nan
            window_squre_sum = np.nan
            dist2end = self.deque_rear - self.deque_front - 1

        start = int(self.deque_rear - dist2end - 1)
        end = int(self.deque_rear - 1)

        # 重新计算参数
        dist2end, window_sum, window_squre_sum, mean_res = update_window_std_params(
            self.deque_timestamp,
            self.deque_vals,
            start, end, window_sum, window_squre_sum, window_size
        )

        # 更新预置参数
        new_params = np.array(
            [dist2end, window_sum, window_squre_sum], dtype=np.float64
        )
        self.deque_stats[field_name] = new_params

        return mean_res

    def get_window_shift(self, n_shift):
        '''抽取当前时刻给定上n_shift个时刻的数据的值'''
        if n_shift > (self.deque_rear - self.deque_front):
            return np.nan
        else:
            return self.deque_vals[self.deque_rear - 1]

    def get_window_range_count(self, window_size, low, high):
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

        # 重新计算参数
        dist2end, low_count, high_count = update_window_range_count_params(
            self.deque_timestamp,
            self.deque_vals,
            start, end, low, high,
            low_count, high_count, window_size
        )

        # 更新预置参数
        new_params = np.array(
            [dist2end, low_count, high_count], dtype=np.float64
        )
        self.deque_stats[field_name] = new_params
        count_res = (dist2end - low_count - high_count) / dist2end

        return count_res

    def get_window_hog_1d(self, window_size, low, high, n_bins):
        '''抽取window_size内的1-D Histogram of Gradient统计量'''
        # 输入检查
        if n_bins <= 0:
            raise ValueError('Invalid n_bins !')
        elif low < -90 or high > 90 or low > high:
            raise ValueError('Invalid low or high value !')
        self.check_window_size(window_size)

        # 载入stream参数（加法hash计算索引）
        field_name = hash(window_size) + hash(n_bins) + hash(low) + hash(high)

        if field_name in self.deque_stats:
            bin_count_meta = self.deque_stats[field_name]
            dist2end, bin_count = bin_count_meta[0], bin_count_meta[1:]
        else:
            bin_count = np.zeros((n_bins, ), dtype=np.float64)
            dist2end = self.deque_rear - self.deque_front - 1

        start = int(self.deque_rear - dist2end - 1)
        end = int(self.deque_rear - 1)

        # 重新计算参数
        dist2end, bin_count = update_window_degree_bin_count_params(
            self.deque_timestamp,
            self.deque_vals,
            start, end, low, high,
            bin_count, self.interval, window_size
        )

        # 更新预置参数
        new_params = np.hstack(
            (np.array([dist2end], dtype=np.float64),
             bin_count.astype(np.float64))
        )
        self.deque_stats[field_name] = new_params
        count_res = bin_count / dist2end

        return count_res

    def get_window_weighted_mean(self, window_size, weight_array):
        '''计算指定window_size内的带权平均值'''
        pass

    def get_exponential_weighted_mean(self, window_size, alpha):
        '''抽取给定window_size内的EWMA加权结果'''
        pass

    def get_window_max(self):
        pass

    def get_window_min(self):
        pass

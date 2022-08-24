# -*- coding: utf-8 -*-

# Created on 202208091516
# Author:    zhuoyin94 <zhuoyin94@163.com>
# Github:    https://github.com/MichaelYin1994

'''
进行KPI异常检测效果评估的Metrics。
'''

import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
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


def feval_custom_metric(threshold):
    '''对于官方Metric的近似（没有依据时间戳的label调节），用于XGBoost的早停'''
    def compute_adjusted_f1(y_pred, dtrain):
        y_true_label = dtrain.get_label()
        y_pred_label = np.where(y_pred > threshold, 1, 0)

        y_pred_label_adjusted = adjust_predict_label(
            y_true_label, y_pred_label, delay=7
        )
        custom_score, _, _ = njit_f1(y_true_label, y_pred_label_adjusted)

        return 'adjusted_f1', -1 * np.round(custom_score, 7)
    return compute_adjusted_f1


def evaluate_df_score(true_df, pred_df, delay=7):
    '''依据比赛[1]与论文[2]的评测方法重构的KPI预测结果的分数计算方法。

    输入类型为pandas DataFrame类型，DataFrame必须包括3列："kpi_id", "label"与
    "timestamp"。其中timestamp列为unix-like时间戳。重构主要是使用LLVM编译加快了分数计算
    速度。

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

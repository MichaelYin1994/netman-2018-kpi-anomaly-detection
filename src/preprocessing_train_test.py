# -*- coding: utf-8 -*-

# Created on 202108311016
# Author:    zhuoyin94 <zhuoyin94@163.com>
# Github:    https://github.com/MichaelYin1994

'''
本模块(create_train_test.py)针对原始*.csv的KPI数据进行预处理，切分训练与测试数据。（注意：test标签应该是靠谱的）
'''

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

from utils.io_utils import LoadSave, load_from_csv

sns.set(style='ticks', font_scale=1.2, palette='deep', color_codes=True)
colors = ['C' + str(i) for i in range(0, 9+1)]

###############################################################################

def plot_single_kpi(time_stamp, kpi_vals, label):
    '''可视化单条KPI曲线'''
    fig, ax = plt.subplots(figsize=(16, 4))

    # plot kpi curve
    ax.plot(time_stamp, kpi_vals,
            linestyle='--', color='k',
            linewidth=1.5, label="KPI curve")

    # plot anomalies
    anomalies_timestamp = time_stamp[label == 1]
    anomalies_pts = kpi_vals[label == 1]
    ax.plot(anomalies_timestamp, anomalies_pts,
            linestyle=' ', markersize=2.5,
            color='red', marker='.', label="Anomaly")

    ax.grid(True)
    ax.legend(fontsize=10)
    ax.set_xlim(np.min(time_stamp), np.max(time_stamp))
    ax.set_ylabel('KPI Value', fontsize=10)
    ax.set_xlabel('Unix timestamp', fontsize=10)
    ax.tick_params(axis='both', labelsize=10)
    plt.tight_layout()


class CONFIGS:
    nrows = None
    is_plot_kpi_curve = True

    test_dev_ratio = 0.3
    train_data_path = '../data/kpi_competition/'


if __name__ == '__main__':
    # 数据预处理
    # *************

    # 1. 按时间切分时序数据为训练与测试部分
    # 2. 拆分测试数据，构建流测试输入
    # ----------------
    train_df = load_from_csv(
        dir_name=CONFIGS.train_data_path, file_name='phase2_train.csv', nrows=CONFIGS.nrows
    )
    test_df = pd.read_hdf(
        os.path.join(CONFIGS.train_data_path, 'phase2_ground_truth.hdf')
    )
    test_df['KPI ID'] = test_df['KPI ID'].apply(str)

    rename_dict = {
        'KPI ID': 'kpi_id', 'timestamp': 'unix_ts'
    }
    train_df.rename(rename_dict, axis=1, inplace=True)
    test_df.rename(rename_dict, axis=1, inplace=True)

    train_df['row_count'] = 1
    train_df.sort_values(by='unix_ts', ascending=True, inplace=True)
    train_df.reset_index(drop=True, inplace=True)
    train_df['unix_ts'] = (train_df['unix_ts'] * 10**3).astype(int)

    test_df['row_count'] = 1
    test_df.sort_values(by='unix_ts', ascending=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    test_df['unix_ts'] = (test_df['unix_ts'] * 10**3).astype(int)

    train_unique_kpis = train_df['kpi_id'].unique().tolist()
    test_unique_kpis = test_df['kpi_id'].unique().tolist()

    # 对train_df与test_df中的kpi_id进行编码
    # *************
    encoder = LabelEncoder()
    encoder.fit(train_df['kpi_id'].values)
    train_df['kpi_id'] = encoder.transform(train_df['kpi_id'].values)
    test_df['kpi_id'] = encoder.transform(test_df['kpi_id'].values)

    # 绘制1条KPI曲线及其异常点（kpi_id \in [0, 28]）
    if CONFIGS.is_plot_kpi_curve:
        kpi_id = 12
        plt.close('all')

        train_tmp_df = train_df[train_df['kpi_id'] == kpi_id]
        plot_single_kpi(
            train_tmp_df['unix_ts'].values,
            train_tmp_df['value'].values,
            train_tmp_df['label'].values
        )

    # 拆分测试数据：按时间戳范围与预设比例，拆分test数据
    # *************

    # test数据按时间顺序与kpi-id拆分为2部分:
    # - test_part_x用做离线测评；
    # - test_part_y用作在线评测
    test_df_list = []
    for kpi_id in test_df['kpi_id'].unique():
        test_df_tmp = test_df.query('kpi_id == {}'.format(kpi_id))
        test_df_tmp.reset_index(drop=True, inplace=True)

        test_df_list.append(test_df_tmp)

    test_idx_list = [
        int(np.floor(CONFIGS.test_dev_ratio * len(item))) for item in test_df_list
    ]

    test_df_list_part_x, test_df_list_part_y = [], []
    for i in range(len(test_idx_list)):
        test_df_list_part_x.append(
            test_df_list[i].iloc[:test_idx_list[i]]
        )
        test_df_list_part_y.append(
            test_df_list[i].iloc[test_idx_list[i]:].reset_index(drop=True)
        )

    test_df_part_x = pd.concat(
        test_df_list_part_x, axis=0, ignore_index=True
    )
    test_df_part_y = pd.concat(
        test_df_list_part_y, axis=0, ignore_index=True
    )

    # 以*.csv保存预处理好的数据
    # ----------------
    train_df.to_csv('../cached_data/train_df.csv', index=False)
    test_df.to_csv('../cached_data/test_df.csv', index=False)
    test_df_part_x.to_csv('../cached_data/test_df_part_x.csv', index=False)
    test_df_part_y.to_csv('../cached_data/test_df_part_y.csv', index=False)

    # 以*.pkl保存预处理好的数据
    # ----------------
    file_handler = LoadSave(dir_name='../cached_data/')
    file_handler.save_data(
        file_name='train_df.pkl', data_file=train_df
    )
    file_handler.save_data(
        file_name='test_df.pkl', data_file=test_df
    )

    file_handler.save_data(
        file_name='test_df_part_x.pkl', data_file=test_df_part_x
    )
    file_handler.save_data(
        file_name='test_df_part_y.pkl', data_file=test_df_part_y
    )

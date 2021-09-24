#!/usr/bin/env python3
#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202109110057
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
本模块(online_inference.py)按离散点的形式，发送流数据，并使用Trained Model进行流式推理。
'''

import gc
import warnings
from datetime import datetime
import multiprocessing as mp

import treelite
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import treelite
import treelite_runtime

from compute_feature_engineering import compute_kpi_feats_single_step
from utils import LoadSave, evaluate_df_score

# 设定全局随机种子，并且屏蔽warnings
GLOBAL_RANDOM_SEED = 2021
np.random.seed(GLOBAL_RANDOM_SEED)
warnings.filterwarnings('ignore')
sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
###############################################################################

if __name__ == '__main__':
    IS_VISUALIZING = False
    KPI_ID_TO_INFERENCE = 12  # kpi_id \in [0. 28]
    TEST_FILE_NAME = 'test_df_part_x.pkl'  # test_df_part_x, test_df_part_y, test_df
    MODEL_FILE_NAME = '15_lgb_nfolds_5_valprauc_424295_valrocauc_912195.pkl'

    # 载入test_df数据
    # ----------------
    file_processor = LoadSave(dir_name='../cached_data/')
    test_df_total = file_processor.load_data(file_name=TEST_FILE_NAME)
    test_df_total = test_df_total.sort_values(by=['timestamp'])
    test_df = test_df_total.query(
        'kpi_id == {}'.format(KPI_ID_TO_INFERENCE)
    ).reset_index(drop=True)

    # 重排特征顺序
    test_df = test_df[['kpi_id', 'timestamp', 'value', 'label']]

    # 载入trained models，并进行编译加速
    # https://treelite.readthedocs.io/en/latest/
    # ----------------
    file_processor = LoadSave(dir_name='../models/')
    trained_models_list, decision_threshold = file_processor.load_data(
        file_name=MODEL_FILE_NAME
    )

    # 载入训练数据的StreamDeque对象
    # ----------------
    file_processor = LoadSave(dir_name='../cached_data/')
    stream_deque_dict = file_processor.load_data(
        file_name='train_stream_deque_dict.pkl'
    )
    stream_deque = stream_deque_dict[KPI_ID_TO_INFERENCE]

    # 模拟按时间戳发送test数据
    # ----------------
    plt.close('all')
    if IS_VISUALIZING:
        plt.ion()
        fig, ax = plt.subplots(figsize=(14, 6))

        ax.grid(True)
        ax.set_ylim(
            np.quantile(test_df['value'].values, 0.005),
            np.quantile(test_df['value'].values, 0.995)
        )
        ax.set_xlabel('Timestamp', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.set_title(
            'Real-time detection on KPI: {}'.format(KPI_ID_TO_INFERENCE),
            fontsize=10)
        ax.tick_params(axis="both", labelsize=10)
        plt.tight_layout()

    predicted_label_list, infer_count, plot_count, prev = [], 0, 0, None
    for data_pack in tqdm(test_df.values):

        # 抽取(timestamp, value)信息与标签信息
        data_pack, true_label = data_pack[:3], data_pack[-1]

        # 数据入队 + 特征抽取
        real_time_feats = compute_kpi_feats_single_step(
            data_pack, stream_deque
        )[1:]
        real_time_feats = np.array(real_time_feats).reshape(1, -1)

        # 模型inference
        test_pred_proba_list_tmp = []
        for fold, model in enumerate(trained_models_list):
            test_pred_proba_list_tmp.append(model.predict_proba(
                real_time_feats, num_iteration=model.best_iteration_, n_jobs=1
            )[:, 1])
            test_pred_label = test_pred_proba_list_tmp[-1] > decision_threshold
        predicted_label_list.append(test_pred_label)

        # 可视化
        if IS_VISUALIZING:
            # First point
            if prev is None:
                prev = data_pack[2]
                infer_count += 1
                continue

            # plot curve
            curr = data_pack[2]

            if true_label == 1 and test_pred_label == 0:
                lines = ax.plot(
                    [infer_count-1, infer_count], [prev, curr],
                    marker='o', markersize=4.5, color='r', linestyle='--'
                )
                plot_count += 1

            prev = curr
            ax.set_xlim(max(0, infer_count - 200), infer_count)
            plt.pause(0.0001)

            if plot_count > 200:
                ax.lines.pop(0)

        infer_count += 1

    # 模拟按时间戳发送test数据
    # ----------------
    plt.close('all')
    fig, ax = plt.subplots(figsize=(18, 6))

    ax.grid(True)
    ax.set_ylim(
        test_df['value'].values.min() - test_df['value'].values.min() * 0.1,
        test_df['value'].values.max() + test_df['value'].values.max() * 0.1
    )
    ax.set_xlim(0, len(test_df))
    ax.set_xlabel('Timestamp', fontsize=10)
    ax.set_ylabel('Value', fontsize=10)
    ax.set_title(
        'Real-time detection on KPI: {}'.format(KPI_ID_TO_INFERENCE),
        fontsize=10)
    ax.tick_params(axis="both", labelsize=10)
    plt.tight_layout()

    # plot kpi curve
    ax.plot(
        np.arange(0, len(test_df)), test_df['value'].values, label='value',
        marker='.', markersize=4.5, color='k', linestyle='--'
    )

    # plot anomalies
    anomalies_idx = np.arange(0, len(test_df))[test_df['label'].values == 1]
    anomalies_pts = test_df['value'].values[test_df['label'].values == 1]
    ax.plot(anomalies_idx, anomalies_pts,
            linestyle=' ', markersize=5.5,
            color='r', marker='s', label="Anomaly")

    # plot predicted anomalies
    predicted_label = np.array(predicted_label_list).astype(int).ravel()
    anomalies_idx = np.arange(0, len(test_df))[predicted_label == 1]
    anomalies_pts = test_df['value'].values[predicted_label == 1]
    ax.plot(anomalies_idx, anomalies_pts,
            linestyle=' ', markersize=4.5,
            color='g', marker='o', label="Predicted Anomaly")

    ax.legend(fontsize=10)

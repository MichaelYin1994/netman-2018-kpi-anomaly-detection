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

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import LoadSave, evaluate_df_score

# 设定全局随机种子，并且屏蔽warnings
GLOBAL_RANDOM_SEED = 2021
np.random.seed(GLOBAL_RANDOM_SEED)
warnings.filterwarnings('ignore')
###############################################################################


if __name__ == '__main__':
    KPI_ID_TO_INFERENCE = 0  # kpi_id \in [0. 28]
    TEST_FILE_NAME = 'test_df_part_x.pkl'  # test_df_part_x, test_df_part_y, test_df

    # 载入test_df数据
    # ----------------
    file_processor = LoadSave(dir_name='../cached_data/')
    test_df = file_processor.load_data(file_name=TEST_FILE_NAME)
    test_df = test_df.sort_values(by=['timestamp'])
    test_df = test_df.query(
        'kpi_id == {}'.format(KPI_ID_TO_INFERENCE)
    ).reset_index(drop=True)

    # 重排特征顺序
    test_df = test_df[['kpi_id', 'timestamp', 'value', 'label']]

    # 模拟按时间戳发送test数据
    # ----------------
    true_label_list, predicted_label_list = [], []
    for data_pack in tqdm(test_df.values):
        # 解压数据值
        kpi_id, timestep, value, label = data_pack

        # 数据入队

        # 统计特征抽取

        # 特征拼装与pre-processing

        # 模型inference

        # post-processing

        # 可视化

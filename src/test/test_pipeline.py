# -*- coding: utf-8 -*-

# Created on 202208301154
# Author:    zhuoyin94 <zhuoyin94@163.com>
# Github:    https://github.com/MichaelYin1994

'''
测试基于flask部署的在线推理pipeline。
'''

import json
import os
import pickle
import sys
import time

sys.path.append('..')

import requests
import pandas as pd
import numpy as np
from easydict import EasyDict
from tqdm import tqdm

from utils.metric_utils import njit_f1
from utils.io_utils import LoadSave
from utils.logger import get_datetime, get_logger
from utils.metric_utils import evaluate_df_score


# 日志配置
# *******************
LOGGING_PATH = '../../logs/'
LOGGING_FILENAME = '{} {}.log'.format(
    get_datetime(), 'test_inference'
)

logger = get_logger(
    logger_name='test_inference',
    is_print_std=True,
    is_send_dingtalk=False,
    is_save_to_disk=False,
    log_path=os.path.join(LOGGING_PATH, LOGGING_FILENAME)
)

# flask server配置信息
# *******************
flask_server_configs = EasyDict()

flask_server_configs.ip = 'localhost'
flask_server_configs.port = '5000'
flask_server_configs.url = 'http://{}:{}/inference_batch'.format(
    flask_server_configs.ip, flask_server_configs.port
)

# 发送测试数据并接收结果
# *******************
if __name__ == '__main__':

    # 数据准备
    # ----------
    file_handler = LoadSave(dir_name='../../cached_data/')
    test_df = file_handler.load_data(file_name='test_df_part_x.pkl')

    # 切分待发送数据
    test_df = test_df.sort_values(by='unix_ts', ascending=True).reset_index(drop=True)

    # 切分待发送数据
    front, rear = 0, 0
    test_df_to_send_list, min_time_interval = [], 30 * 60

    ts_arr = (test_df['unix_ts'].values) / 10**3
    while(rear < len(test_df)):
        if rear == len(test_df) - 1:
            test_df_to_send_list.append(test_df.iloc[front:])
            rear += 1
        else:
            if (ts_arr[rear] - ts_arr[front]) < min_time_interval:
                rear += 1
            else:
                test_df_to_send_list.append(test_df.iloc[front:rear])
                front = rear

    # 发送数据到推理服务器，获取推理结果
    # ----------
    test_df_to_send_list = test_df_to_send_list[:4096]

    pred_label_list = []
    true_label_list = []

    for df in tqdm(test_df_to_send_list):
        data_pack = {'dataframe': df}
        data_to_send = pickle.dumps(data_pack)
        response = requests.post(url=flask_server_configs.url, data=data_to_send)

        # 将json/pickle数据parse
        pred_label_list.extend(response.json()['class_id_list'])
        true_label_list.extend(df['label'].values.tolist())

    print('Pred label count: {}'.format(np.bincount(pred_label_list)))
    print('True label count: {}'.format(np.bincount(true_label_list)))
    print(
        njit_f1(np.array(true_label_list), np.array(pred_label_list))
    )
    print('Total record count: {}'.format(sum([len(item) for item in test_df_to_send_list])))

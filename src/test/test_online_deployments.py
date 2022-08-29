# -*- coding: utf-8 -*-

# Created on 202208261034
# Author:    zhuoyin94 <zhuoyin94@163.com>
# Github:    https://github.com/MichaelYin1994

'''
测试在线部署的feature server与inference server。
'''

import json
import os
import pickle
import sys
import time

sys.path.append('..')

import numpy as np
import pandas as pd
import requests
import tritonclient.grpc as triton_grpc
from flask import Flask, Response, request
from tqdm import tqdm
from tritonclient import utils as triton_utils

from utils.configs import fe_server_configs as CONFIGS
from utils.io_utils import LoadSave
from utils.logger import get_datetime, get_logger
from utils.metric_utils import evaluate_df_score

# 日志配置
# *******************
LOGGING_PATH = '../logs/'
LOGGING_FILENAME = '{} {}.log'.format(
    get_datetime(), CONFIGS.task_name
)

logger = get_logger(
    logger_name=CONFIGS.task_name,
    is_print_std=True,
    is_send_dingtalk=False,
    is_save_to_disk=False,
    log_path=os.path.join(LOGGING_PATH, LOGGING_FILENAME)
)

# 测试数据载入
# *******************
file_handler = LoadSave(dir_name='../../cached_data/')
test_df = file_handler.load_data(file_name='test_df_part_x.pkl')

test_df = test_df.iloc[:4096].reset_index(drop=True)

# 将数据类型进行转换
test_df_dict = test_df.to_dict(orient='list')
col_name_list = list(test_df.columns)

# Feature server测试
# *******************
# curl http://127.0.0.1:9080/dbs/kpi_data/deployments/realtime_feats_service -X POST -d'{"input": [[1471044900000, -1.43640797338, 0, 20, 1]]}'
# data = {
#     "input": [[1471044900000, -1.43640797338, 0, 20, 1], [1471044900000, -1.43640797338, 0, 20, 1]]
# }
url_fe_server = 'http://{}:{}/dbs/{}/deployments/{}'.format(
    CONFIGS.zk_ip, CONFIGS.port, CONFIGS.db_name, CONFIGS.task_name
)
headers = {'Content-Type': 'application/json'}

bs, total_size = 4, len(test_df)
for i in tqdm(range(0, len(test_df), bs)):
    data = {'input': []}

    for j in range(bs):
        tmp = []
        for col_name in col_name_list:
            tmp.append(test_df_dict[col_name][min(i+j, total_size-1)])
        data['input'].append(tmp)

    response = requests.post(
        url_fe_server, data=json.dumps(data), headers=headers
    ).json()

# Inference server测试
# *******************

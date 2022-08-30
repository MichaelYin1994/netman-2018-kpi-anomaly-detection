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
import gc

sys.path.append('..')

import numpy as np
import pandas as pd
import requests
import tritonclient.grpc as triton_grpc
from flask import Flask, Response, request
from tqdm import tqdm
from tritonclient import utils as triton_utils

from utils.configs import fe_server_configs, infer_server_configs
from utils.io_utils import LoadSave, YamlParser
from utils.logger import get_datetime, get_logger
from utils.metric_utils import evaluate_df_score
from utils.triton_utils import get_grpc_triton_predict_proba

# 日志配置
# *******************
LOGGING_PATH = '../logs/'
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

# 测试数据载入
# *******************
file_handler = LoadSave(dir_name='../../cached_data/')
test_df = file_handler.load_data(file_name='test_df_part_x.pkl')

test_df = test_df.iloc[:32].reset_index(drop=True)

# 将数据类型进行转换
test_df_dict = test_df.to_dict(orient='list')
col_name_list = list(test_df.columns)

# 载入fe_server_configs配置文件
yaml_parser = YamlParser(config_file='../../cached_data/train_feats_meta/feature_meta.yaml')
fe_server_configs['feat_names'] = yaml_parser.config_dict['feat_names']

# 载入infer_server_configs配置文件
yaml_parser = YamlParser(config_file='../../cached_models/xgb_gpu_models/xgboost_configs.yaml')
infer_server_configs['n_features'] = yaml_parser.config_dict['n_feats']
infer_server_configs['n_classes'] = yaml_parser.config_dict['n_classes']

# Feature server测试
# *******************
# curl http://127.0.0.1:9080/dbs/kpi_data/deployments/realtime_feats_service -X POST -d'{"input": [[1471044900000, -1.43640797338, 0, 20, 1]]}'
# data = {
#     "input": [[1471044900000, -1.43640797338, 0, 20, 1], [1471044900000, -1.43640797338, 0, 20, 1]]
# }
# 将数据类型进行转换
test_df_dict = test_df.to_dict(orient='list')
col_name_list = list(test_df.columns)

total_size, res_list = len(test_df), []
for i in tqdm(range(total_size)):
    data = {'input': []}

    # 构建样本
    sample = []
    for col_name in col_name_list:
        sample.append(test_df_dict[col_name][i])
    data['input'].append(sample)

    # 发送样本获取结果
    response = requests.post(
        fe_server_configs.url,
        data=json.dumps(data),
        headers=fe_server_configs.headers
    ).json()

    # 保存推理结果
    res_list.append(response)

test_feats_vals = [
    item['data']['data'][0] for item in res_list
]
test_feats_df = pd.DataFrame(
    test_feats_vals,
    columns=fe_server_configs['feat_names']
)

print(test_feats_df.head(4))

# Inference server测试
# *******************
triton_client = triton_grpc.InferenceServerClient(
    url=infer_server_configs.url, verbose=False
)

test_pred_proba_list = []
for model_version in infer_server_configs.model_version_list:
    test_pred_proba_list.append(
        get_grpc_triton_predict_proba(
            triton_client,
            infer_server_configs.model_name,
            model_version,
            test_feats_df.drop(fe_server_configs['key_feats'], axis=1).values.astype(np.float32)
        )
    )

test_pred_avg_proba = np.mean(
    test_pred_proba_list, axis=0
)
print(test_pred_avg_proba)

del triton_client

print('**********************DONE**********************')

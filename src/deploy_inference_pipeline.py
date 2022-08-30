# -*- coding: utf-8 -*-

# Created on 202208291113
# Author:    zhuoyin94 <zhuoyin94@163.com>
# Github:    https://github.com/MichaelYin1994

'''
基于flask部署openmldb + triton infer server的pipeline。
'''

import json
import os
import pickle
import sys
import time

sys.path.append('..')

import logging
import requests
import numpy as np
import pandas as pd
import tritonclient.grpc as triton_grpc
from flask import Flask, Response, request
from tritonclient import utils as triton_utils

from utils.configs import fe_server_configs, infer_server_configs
from utils.io_utils import YamlParser
from utils.logger import get_datetime, get_logger
from utils.metric_utils import evaluate_df_score
from utils.triton_utils import get_grpc_triton_predict_proba

app = Flask(__name__)
log = logging.getLogger("werkzeug")
log.setLevel(logging.WARNING)

# 载入fe_server_configs配置文件
yaml_parser = YamlParser(config_file='../cached_data/train_feats_meta/feature_meta.yaml')
fe_server_configs['feat_names'] = yaml_parser.config_dict['feat_names']

# 载入infer_server_configs配置文件
yaml_parser = YamlParser(config_file='../cached_models/xgb_gpu_models/xgboost_configs.yaml')
infer_server_configs['n_features'] = yaml_parser.config_dict['n_feats']
infer_server_configs['n_classes'] = yaml_parser.config_dict['n_classes']

# 生成logger实例
LOGGING_PATH = '../logs/'
LOGGING_FILENAME = '{} {}.log'.format(
    get_datetime(), 'pipeline_server'
)

logger = get_logger(
    logger_name='pipeline_server',
    is_print_std=True,
    is_send_dingtalk=False,
    is_save_to_disk=False,
    log_path=os.path.join(LOGGING_PATH, LOGGING_FILENAME)
)

# 数据库连接


# triton client
triton_client = triton_grpc.InferenceServerClient(
    url=infer_server_configs.url, verbose=False
)


def preprocessing(df):
    '''输入数据预处理'''
    return df

def create_cross_feats(df):
    '''基于window特征构建交叉特征群'''
    return df

def create_openmldb_feats_with_insertion(df):
    '''基于openmldb构建window上的特征工程'''

    # 数据预处理
    df_dict = df.to_dict(orient='list')
    col_name_list = list(df.columns)

    # 发送到openmldb获取window计算结果
    total_size, response_list = len(df), []
    for i in range(total_size):
        data = {'input': []}

        # 构建样本
        sample = []
        for col_name in col_name_list:
            sample.append(df_dict[col_name][i])
        data['input'].append(sample)

        # 发送样本获取结果
        response = requests.post(
            fe_server_configs.url,
            data=json.dumps(data),
            headers=fe_server_configs.headers
        ).json()

        # 插入数据到openmldb
        sql_statement = 'INSERT INTO '

        # 保存推理结果
        response_list.append(response)

    feat_arr = [
        item['data']['data'][0] for item in response_list
    ]
    feat_df = pd.DataFrame(feat_arr, columns=fe_server_configs['feat_names'])

    # window特征交叉
    feat_df = create_cross_feats(feat_df)

    return feat_df


def postprocessing(pred_res_list):
    '''推理结果后处理'''
    pred_avg_proba = np.mean(pred_res_list, axis=0)
    pred_label = np.argmax(pred_avg_proba, axis=1)

    return pred_label


@app.route('/inference_batch', methods=['POST'])
def inference_batch():
    '''后端推理pipeline'''

    # 将json/pickle数据parse为pandas DataFrame形式
    datapack_dict = pickle.loads(request.data)
    test_df = datapack_dict['dataframe']

    # 前处理
    # ----------
    test_df = preprocessing(test_df)

    # 逐条数据：openmldb窗口特征工程 && 插入数据到openmldb在线表中
    # ----------
    test_feat_df = create_openmldb_feats_with_insertion(test_df)
    test_feat_vals = test_feat_df.drop(fe_server_configs['key_feats'], axis=1).values.astype(np.float32)

    # 发送到Infer server进行推理
    # ----------
    test_pred_proba_list = []
    for model_version in infer_server_configs.model_version_list:
        test_pred_proba_list.append(
            get_grpc_triton_predict_proba(
                triton_client,
                infer_server_configs.model_name,
                model_version,
                test_feat_vals
            )
        )

    # 后处理
    # ----------
    test_pred_label = postprocessing(test_pred_proba_list)

    # 返回处理结果
    # ----------
    ret_dict = {'class_id_list': test_pred_label.tolist()}

    return Response(json.dumps(ret_dict), mimetype='application/json')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

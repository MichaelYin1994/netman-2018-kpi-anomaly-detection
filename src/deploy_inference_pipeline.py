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

import numpy as np
import pandas
import tritonclient.grpc as triton_grpc
from flask import Flask, Response, request
from tritonclient import utils as triton_utils

from utils.configs import fe_server_configs, infer_server_configs
from utils.logger import get_datetime, get_logger
from utils.metric_utils import evaluate_df_score
from utils.triton_utils import get_grpc_triton_predict_proba

app = Flask(__name__)

# 生成logger实例

def preprocessing(test_df):
    '''输入数据预处理'''
    return test_df

def create_cross_feats():
    '''基于window特征构建交叉特征群'''
    return

def create_openmldb_feats_with_insertion(test_df):
    '''基于openmldb构建window上的特征工程'''
    size

    # 发送到openmldb获取window计算结果

    # 插入数据到数据库

    # window特征交叉
    create_cross_feats()

    # 返回特征工程结果

    pass

def postprocessing():
    '''推理结果后处理'''
    pass

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
    test_feats_df = create_openmldb_feats_with_insertion(test_df)

    # 发送到Infer server进行推理
    # ----------
    test_pred_proba_list = []
    for model_version in infer_server_configs.model_version_list:
        pass

    # 后处理
    # ----------
    postprocessing()

    # 返回处理结果
    # ----------

    # return Response(json.dumps(ret_dict),  mimetype='application/json')
    return


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

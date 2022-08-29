# -*- coding: utf-8 -*-

# Created on 202208261532
# Author:    zhuoyin94 <zhuoyin94@163.com>
# Github:    https://github.com/MichaelYin1994

'''
生成Triton Inference Server的XGBoost部署配置文件。
'''

import os

from utils.configs import infer_server_configs as CONFIGS
from utils.io_utils import YamlParser
from utils.triton_utils import generate_xgboost_deployment_config

if __name__ == '__main__':
    # Step 0: 载入模型基础信息文件
    yaml_parser = YamlParser(
        config_file=os.path.join(CONFIGS.model_path, 'xgboost_configs.yaml')
    )
    CONFIGS.n_classes = yaml_parser.config_dict.n_classes
    CONFIGS.n_features = yaml_parser.config_dict.n_feats

    # Step 1: 为模型生成配置文件
    config_text = generate_xgboost_deployment_config(
        n_features=CONFIGS.n_features, n_classes=CONFIGS.n_classes
    )

    # Step 2: 配置文件写到本地
    config_path = os.path.join(CONFIGS.model_path, 'config.pbtxt')
    with open(config_path, 'w') as f:
        f.write('\n'.join(config_text))

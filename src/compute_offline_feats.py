# -*- coding: utf-8 -*-

# Created on 202208251118
# Author:    zhuoyin94 <zhuoyin94@163.com>
# Github:    https://github.com/MichaelYin1994

'''
构建并保存离线特征工程结果。
'''


import os

import pandas as pd
import sqlalchemy as db

from utils.configs import offline_fe_configs as CONFIGS
from utils.configs import openmldb_configs
from utils.io_utils import YamlParser
from utils.logger import get_datetime, get_logger


def create_openmldb_connection(configs):
    '''返回一个数据库的连接实例'''
    engine = db.create_engine(
        'openmldb:///{}?zk={}&zkPath={}'.format(
            configs.db_name,
            '{}:{}'.format(configs.zk_ip, configs.zk_port),
            configs.zk_path
        )
    )
    connection = engine.connect()

    return connection


if __name__ == '__main__':
    # 配置日志logger
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

    # 目录数据预处理
    # *******************

    # 读取特征工程脚本
    # ----------
    with open('sql_feature_engineering.sql', 'r') as f:
        sql_script = f.readlines()

    sql_offline_script = ' '.join([item.strip() for item in sql_script])

    # 构建数据库连接
    # ----------
    connection = create_openmldb_connection(openmldb_configs)

    connection.execute('USE {};'.format(CONFIGS.db_name))
    connection.execute("SET @@execute_mode='offline';")
    connection.execute('SET @@sync_job=true;')
    connection.execute('SET @@job_timeout=1200000;')
    connection.execute(sql_offline_script)

    connection.close()

    # 存储特征表基础信息
    # ----------
    f_dir = '../cached_data/train_feats/'
    target_dir = '../cached_data/train_feats_meta/'

    f_names_list = [f_name for f_name in os.listdir(f_dir) if f_name.endswith('.csv')]
    col_names = list(pd.read_csv(os.path.join(f_dir, f_names_list[0])).columns)

    os.makedirs(target_dir, exist_ok=True)
    config_dict = {'feat_names': col_names}
    yaml_parser = YamlParser(config_dict=config_dict)
    yaml_parser.save(dir=target_dir, f_name='feature_meta.yaml')

    logger.info('\n***************FINISHED...***************')

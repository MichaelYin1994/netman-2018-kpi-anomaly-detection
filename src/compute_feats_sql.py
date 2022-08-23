# -*- coding: utf-8 -*-

# Created on 202208181043
# Author:    zhuoyin94 <zhuoyin94@163.com>
# Github:    https://github.com/MichaelYin1994

'''
使用sql脚本，进行时序特征工程。
'''

import os

import pandas as pd
import sqlalchemy as db
import tqdm

from utils.logger import get_datetime, get_logger


class CONFIGS:
    # 任务名称
    task_name = 'create_feats'

    # 数据库名字
    db_name = 'kpi_data'
    table_name = 'kpi_history_series'

    # zookeeper IP
    # zk_ip = '172.48.0.28'
    zk_ip = '127.0.0.1'

    # zookeeper port
    zk_port = '2181'

    # zookeeper path（默认）
    zk_path = '/openmldb'


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

    # # 数据目录预处理
    # # ----------
    # data_file_list = ['train_feats_df.csv']
    # data_path = '../cached_data/'

    # for f_name in data_file_list:
    #     if f_name in os.listdir(data_path):
    #         os.remove(os.path.join(data_path, f_name))

    # 读取特征工程脚本
    # ----------
    with open('sql_feature_engineering.sql', 'r') as f:
        sql_script = f.readlines()
    sql_script = ' '.join([item.strip() for item in sql_script])

    # 构建数据库连接
    # ----------
    connection = create_openmldb_connection(CONFIGS)

    connection.execute('USE {};'.format(CONFIGS.db_name))
    connection.execute("SET @@execute_mode='offline';")
    connection.execute('SET @@sync_job=true;')
    connection.execute('SET @@job_timeout=1200000;')
    connection.execute(sql_script)

    # # 构建数据库连接
    # # ----------
    # if CONFIGS.is_deploy:
    #     deploy_sql_script = 'DEPLOY compute_feats ' + sql_script

    connection.close()

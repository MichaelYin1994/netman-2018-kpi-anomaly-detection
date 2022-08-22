# -*- coding: utf-8 -*-

# Created on 202208181043
# Author:    zhuoyin94 <zhuoyin94@163.com>
# Github:    https://github.com/MichaelYin1994

'''
使用sql脚本，进行时序特征工程。
'''

import os

import openmldb.dbapi
import pandas as pd
import sqlalchemy as db
import tqdm


class CONFIGS:
    # 是否进行部署
    is_deploy = False

    # 数据库名字
    db_name = 'test_kpi_series'
    table_name = 'kpi_history_series'

    # host机器地址(tct-cvpr, zhuoyin94)
    # db_ip = '172.16.216.122'
    # db_ip = '172.48.0.28'
    db_ip = '127.0.0.1'

    # openmldb nameserver ip port地址
    db_port = '6527'


def create_openmldb_connection(configs):
    '''返回一个数据库的连接实例'''
    connection = openmldb.dbapi.connect(
        db=configs.db_name,
        host=configs.db_ip,
        port=configs.db_port,
    )
    return connection


if __name__ == '__main__':
    # 数据目录预处理
    # ----------
    data_file_list = [
        'train_feats_df.csv'
    ]
    data_path = '../cached_data/'

    for f_name in data_file_list:
        if f_name in os.listdir(data_path):
            os.remove(os.path.join(data_path, f_name))

    # 读取特征工程脚本
    # ----------
    with open('sql_feature_engineering.sql', 'r') as f:
        sql_script = f.readlines()
    sql_script = ' '.join([item.strip() for item in sql_script])

    # 构建数据库连接
    # ----------
    connection = create_openmldb_connection(CONFIGS)
    cursor = connection.cursor()

    cursor.execute('USE {};'.format(CONFIGS.db_name))
    cursor.execute(sql_script)

    # 构建数据库连接
    # ----------
    if CONFIGS.is_deploy:
        deploy_sql_script = 'DEPLOY compute_feats ' + sql_script

    cursor.close()
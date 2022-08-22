# -*- coding: utf-8 -*-

# Created on 202208091620
# Author:    zhuoyin94 <zhuoyin94@163.com>
# Github:    https://github.com/MichaelYin1994

'''
将kpi数据写入OpenMLDB内存数据库。

样本数据:
      unix_ts     value  label                                kpi_id
0  1476460800  0.012604      0  da10a69f-d836-3baa-ad40-3e548ecf1fbd
1  1476460860  0.017786      0  da10a69f-d836-3baa-ad40-3e548ecf1fbd
2  1476460920  0.012014      0  da10a69f-d836-3baa-ad40-3e548ecf1fbd
3  1476460980  0.017062      0  da10a69f-d836-3baa-ad40-3e548ecf1fbd
'''

import numpy as np
import openmldb.dbapi
import pandas as pd
import sqlalchemy as db
import tqdm

from utils.io_utils import LoadSave


class CONFIGS:
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
    # 载入train kpi数据
    # *******************
    file_handler = LoadSave(dir_name='../cached_data/')
    train_df = file_handler.load_data(file_name='train_df.pkl')

    train_df['time_diff'] = train_df.groupby(['kpi_id'])['unix_ts'].diff()
    train_df['time_diff'].fillna(
        train_df['time_diff'].mode().values[0], inplace=True
    )
    train_df['time_diff'] = train_df['time_diff'].astype(int)
    train_df['row_count'] = 1

    train_df.sort_values(by='unix_ts', ascending=True, inplace=True)
    train_df.reset_index(drop=True, inplace=True)

    # train_df = train_df.groupby(['kpi_id']).head(10).reset_index(drop=True)
    train_df['unix_ts'] = (train_df['unix_ts'] * 10**3).astype(int)

    # 数据库操作
    # *******************

    # STEP 1: 创建数据库
    # ----------
    connection = create_openmldb_connection(CONFIGS)
    cursor = connection.cursor()

    sql_statement = 'CREATE DATABASE IF NOT EXISTS {};'.format(CONFIGS.db_name)

    cursor.execute(sql_statement)
    cursor.close()

    # STEP 2: 创建表（设置offline mode）
    # ----------
    connection = create_openmldb_connection(CONFIGS)
    cursor = connection.cursor()

    cursor.execute('USE {};'.format(CONFIGS.db_name))
    cursor.execute(
        (
            'CREATE TABLE IF NOT EXISTS kpi_history_series'
            ' (unix_ts timestamp, value double, label int, kpi_id int, time_diff int, row_count int);'
        )
    )

    # # 使用SQL语句进行数据的loading
    # file_path = '/work/kpi-anomaly-detection/cached_data/train_df.csv'
    # cursor.execute(
    #     (
    #         f"LOAD DATA INFILE '{file_path}'"
    #         "INTO TABLE {} OPTIONS(format='csv',header=true);".format(CONFIGS.table_name)
    #     )
    # )

    cursor.close()

    # STEP 3: 将数据写入OpenMLDB数据库中
    # ----------
    engine = db.create_engine(
        'openmldb:///{}?host={}&port={}'.format(
            CONFIGS.db_name, CONFIGS.db_ip, CONFIGS.db_port
        )
    )
    connection = engine.connect()
    connection.execute('USE {}'.format(CONFIGS.db_name))
    # connection.execute('INSERT INTO {} values(1476460800, 0.0126036806477, 0, 25);'.format(CONFIGS.table_name))

    train_df.to_sql(
        CONFIGS.table_name, con=engine, if_exists='append', index=False
    )
    connection.close()

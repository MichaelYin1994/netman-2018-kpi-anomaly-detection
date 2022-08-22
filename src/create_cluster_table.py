# -*- coding: utf-8 -*-

# Created on 202208091620
# Author:    zhuoyin94 <zhuoyin94@163.com>
# Github:    https://github.com/MichaelYin1994

'''
将kpi数据写入OpenMLDB内存数据库。

样本数据:
    timestamp     value  label                                KPI ID
0  1476460800  0.012604      0  da10a69f-d836-3baa-ad40-3e548ecf1fbd
1  1476460860  0.017786      0  da10a69f-d836-3baa-ad40-3e548ecf1fbd
2  1476460920  0.012014      0  da10a69f-d836-3baa-ad40-3e548ecf1fbd
3  1476460980  0.017062      0  da10a69f-d836-3baa-ad40-3e548ecf1fbd
'''

import pandas as pd
import openmldb.dbapi
import sqlalchemy as db
import tqdm

from utils.io_utils import LoadSave

class CONFIGS:
    # 数据库名字
    db_name = 'test_kpi_series'
    table_name = 'kpi_history_series'

    # zookeeper IP
    zk_ip = '172.48.0.28'

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

    train_df = train_df.groupby(['kpi_id']).head(10).reset_index(drop=True)
    train_df['unix_ts'] = (train_df['unix_ts'] * 10**3).astype(int)

    # 构建zookeeper连接
    # *******************
    connection = create_openmldb_connection(CONFIGS)

    # 创建数据库
    sql_statement = 'CREATE DATABASE IF NOT EXISTS {};'.format(CONFIGS.db_name)
    connection.execute(sql_statement)

    # 创建数据表
    connection.execute('USE {};'.format(CONFIGS.db_name))
    connection.execute("SET @@execute_mode='offline';")
    connection.execute(
        (
            f'CREATE TABLE IF NOT EXISTS {CONFIGS.table_name}'
            ' (unix_ts timestamp, value double, label int, kpi_id int, time_diff int, row_count int);'
        )
    )

    # STEP 3: 将数据写入OpenMLDB数据库中
    # ----------
    connection.execute("SET @@execute_mode='online';")
    # connection.execute('INSERT INTO {} values(1476460800, 0.0126036806477, 0, 25);'.format(CONFIGS.table_name))

    train_df.to_sql(
        CONFIGS.table_name, con=connection, if_exists='append', index=False
    )
    connection.close()

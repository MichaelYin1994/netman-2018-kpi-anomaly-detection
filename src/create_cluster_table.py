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

import os

import pandas as pd
import sqlalchemy as db
import tqdm

from utils.logger import get_datetime, get_logger


class CONFIGS:
    # 任务名称
    task_name = 'create_db'

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

    # offline data path
    offline_data = '/work/kpi-anomaly-detection/cached_data/train_df.csv'

    # online data path
    online_data = '/work/kpi-anomaly-detection/cached_data/test_df_part_x.csv'


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

    # 构建zookeeper连接
    # *******************
    connection = create_openmldb_connection(CONFIGS)

    # 创建数据库
    sql_statement = 'CREATE DATABASE IF NOT EXISTS {};'.format(CONFIGS.db_name)
    connection.execute(sql_statement)

    # 创建数据表
    connection.execute('USE {};'.format(CONFIGS.db_name))
    connection.execute(
        (
            f'CREATE TABLE IF NOT EXISTS {CONFIGS.table_name}'
            ' (unix_ts timestamp, value double, label int, kpi_id int, row_count int);'
        )
    )

    # offline模式下，离线数据载入LOAD DATA
    # *******************
    logger.info('Loading offline data...')

    connection.execute("SET @@execute_mode='offline';")
    connection.execute('SET @@sync_job=true;')
    connection.execute('SET @@job_timeout=1200000;')
    connection.execute(
        (
            f"LOAD DATA INFILE '{CONFIGS.offline_data}' "
            f"INTO TABLE {CONFIGS.table_name} OPTIONS(format='csv', header=true, deep_copy=true, mode='append');"
        )
    )

    # online模式下，离线数据载入LOAD DATA
    # *******************
    logger.info('Loading online data...')

    connection.execute("SET @@execute_mode='online';")
    connection.execute('SET @@sync_job=true;')
    connection.execute('SET @@job_timeout=1200000;')
    connection.execute(
        (
            f"LOAD DATA INFILE '{CONFIGS.offline_data}' "
            f"INTO TABLE {CONFIGS.table_name} OPTIONS(format='csv', header=true, deep_copy=true, mode='append');"
        )
    )

    connection.close()

    # # STEP 3: 将数据写入OpenMLDB数据库中
    # # ----------
    # connection.execute("SET @@execute_mode='online';")
    # train_df.to_sql(
    #     CONFIGS.table_name, con=connection, if_exists='append', index=False
    # )
    # connection.close()

# -*- coding: utf-8 -*-

# Created on 202208181043
# Author:    zhuoyin94 <zhuoyin94@163.com>
# Github:    https://github.com/MichaelYin1994

'''
使用sql脚本，进行时序特征工程。

在线部署联通测试脚本：
curl http://127.0.0.1:9080/dbs/kpi_data/deployments/online_feats -X POST -d'{"input": [[1467303000000, 0.0282821271162999, 0, 20, 1]]}'
'''

import os

import sqlalchemy as db

from utils.configs import online_fe_configs as CONFIGS
from utils.configs import openmldb_configs
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

    sql_online_script = ' '.join([item.strip() for item in sql_script[:-1]])
    connection = create_openmldb_connection(openmldb_configs)

    # 部署SQL脚本并载入online数据
    # ----------

    # 执行部署
    logger.info('Start deployment...')

    connection.execute('USE {};'.format(CONFIGS.db_name))
    connection.execute("SET @@execute_mode='online';")
    connection.execute('SET @@sync_job=true;')
    connection.execute('SET @@job_timeout=1200000;')
    # connection.execute('DROP DEPLOYMENT IF EXISTS {};'.format(CONFIGS.task_name))

    # 创建online数据表
    connection.execute(
        (
            f'CREATE TABLE IF NOT EXISTS {CONFIGS.table_name}'
            ' (unix_ts timestamp, value double, label int, kpi_id int, row_count int);'
        )
    )

    # 部署sql脚本
    deploy_sql_script = f'DEPLOY {CONFIGS.task_name} ' + sql_online_script
    connection.execute(deploy_sql_script)

    # 载入online数据
    connection.execute(
        (
            f"LOAD DATA INFILE '{CONFIGS.online_data}' "
            f"INTO TABLE {CONFIGS.table_name} OPTIONS(format='csv', header=true, deep_copy=true, mode='append');"
        )
    )
    connection.close()

    logger.info('\n***************FINISHED...***************')

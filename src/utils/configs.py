# -*- coding: utf-8 -*-

# Created on 202208251059
# Author:    zhuoyin94 <zhuoyin94@163.com>
# Github:    https://github.com/MichaelYin1994

'''
全局配置类信息。
'''

from easydict import EasyDict

# openmldb server配置项
# *******************
openmldb_configs = EasyDict()

# zookeeper相关配置项
openmldb_configs.zk_ip = '127.0.0.1'
openmldb_configs.zk_port = '2181'
openmldb_configs.zk_path = '/openmldb'
openmldb_configs.db_name = 'kpi_data'

# offline数据导入配置项
# *******************
offline_load_configs = EasyDict()

# 任务配置项
offline_load_configs.task_name = 'load_offline_data'
offline_load_configs.db_name = openmldb_configs.db_name
offline_load_configs.table_name = 'offline_kpi_history_series'

# offline数据路径
offline_load_configs.offline_data = '/work/kpi-anomaly-detection/cached_data/train_df.csv'

# offline特征工程配置项
# *******************
offline_fe_configs = EasyDict()

# 任务配置项
offline_fe_configs.task_name = 'create_offline_kpi_feats' 
offline_fe_configs.db_name = openmldb_configs.db_name
offline_fe_configs.table_name = 'offline_kpi_history_series'

# online特征工程配置项
# *******************
online_fe_configs = EasyDict()

# 任务配置项
online_fe_configs.task_name = 'realtime_feats_service'
online_fe_configs.db_name = openmldb_configs.db_name
online_fe_configs.table_name = 'online_kpi_history_series'

# online数据路径
online_fe_configs.online_data = '/work/kpi-anomaly-detection/cached_data/train_df.csv'

# xgb训练配置项
# *******************
xgb_configs = EasyDict()

# 全局随机种子
xgb_configs.global_random_seed = 2077

# 训练相关参数
xgb_configs.task_name = 'train_xgb'
xgb_configs.n_folds = 7
xgb_configs.early_stopping_rounds = 2000
xgb_configs.threshold = 0.5
xgb_configs.xgb_params = {
    'n_estimators': 15000, # 15000
    'max_depth': 4,
    'learning_rate': 0.05,
    'verbosity': 0,
    'objective': 'binary:logistic',
    'booster': 'gbtree',
    'colsample_bytree': 0.85,
    'colsample_bylevel': 0.85,
    'disable_default_eval_metric': 1,
    'subsample': 0.95,
    'random_state': xgb_configs.global_random_seed
}

# 日志及设备配置参数
xgb_configs.gpu_id = 0
xgb_configs.verbose_rounds = 1000
xgb_configs.is_save_log_to_disk = False
xgb_configs.is_save_model_to_disk = True

# Feature server配置项
# *******************
fe_server_configs = EasyDict()

# 任务配置项
fe_server_configs.port = '9080'
fe_server_configs.db_name = online_fe_configs.db_name
fe_server_configs.url = 'http://{}:{}/dbs/{}/deployments/{}'.format(
    openmldb_configs.zk_ip, fe_server_configs.port, online_fe_configs.db_name, online_fe_configs.task_name
)
fe_server_configs.headers = {'Content-Type': 'application/json'}
fe_server_configs.key_feats = ['kpi_id', 'unix_ts', 'label']

# Inference server配置项
# *******************
infer_server_configs = EasyDict()

infer_server_configs.n_features = 1927
infer_server_configs.n_classes = 2
infer_server_configs.model_path = '../cached_models/xgb_gpu_models'

infer_server_configs.ip = 'localhost'
infer_server_configs.port = '8001'
infer_server_configs.url = '{}:{}'.format(
    infer_server_configs.ip, infer_server_configs.port
)
infer_server_configs.n_classes = 2
infer_server_configs.model_name = 'xgb_gpu_models'
infer_server_configs.model_version_list = ['6', '7']

# Flask server配置项
# *******************

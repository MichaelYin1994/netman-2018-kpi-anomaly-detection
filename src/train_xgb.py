# -*- coding: utf-8 -*-

# Created on 202203072137
# Author:    zhuoyin94 <zhuoyin94@163.com>
# Github:    https://github.com/MichaelYin1994

'''
训练XGBoost分类器，使用测试集来训练分类器，在训练集全集上验证。
'''

import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score

from metrics.metrics import evaluate_df_score, feval_custom_metric
from utils.io_utils import serialize_xgboost_model
from utils.logger import get_datetime, get_logger

gpus = tf.config.experimental.list_physical_devices('GPU')

class CONFIGS:
    n_folds = 5
    task_name = 'train_xgb'
    early_stopping_rounds = 2000
    threshold = 0.5

    gpu_id = 0
    verbose_rounds = 1000
    global_random_seed = 2077
    is_save_log_to_disk = True
    is_save_model_to_disk = False

    xgb_params = {
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
        'random_state': global_random_seed
    }


def generator_tscv(df_list, n_folds=2, is_shuffle_train=True):
    '''Generator of the data'''
    n_splits = n_folds + 1
    df_size_list = [len(item) for item in df_list]

    for i in range(n_folds):
        train_idx_list = [
            int(np.floor(item / n_splits * (i + 1))) for item in df_size_list
        ]
        valid_idx_list = [
            int(np.ceil(item / n_splits * (i + 2))) for item in df_size_list
        ]

        df_train_list, df_valid_list = [], []
        for j, df in enumerate(df_list):
            df_train_list.append(
                df.iloc[:train_idx_list[j]]
            )
            df_valid_list.append(
                df.iloc[train_idx_list[j]:valid_idx_list[j]]
            )

        df_train = pd.concat(
            df_train_list, axis=0, ignore_index=True
        )
        df_valid = pd.concat(
            df_valid_list, axis=0, ignore_index=True
        )

        # 通过sampling的方式对DataFrame进行随机shuffle
        if is_shuffle_train:
            df_train = df_train.sample(
                frac=1, random_state=CONFIGS.global_random_seed
            )
            df_train.reset_index(drop=True, inplace=True)

        yield i, df_train, df_valid


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(CONFIGS.global_random_seed)


if __name__ == '__main__':
    # 全局化的参数
    # *******************

    # 配置日志格式
    # ----------
    LOGGING_PATH = '../logs/'
    LOGGING_FILENAME = '{} {}.log'.format(
        get_datetime(), CONFIGS.task_name
    )

    logger = get_logger(
        logger_name=CONFIGS.task_name,
        is_print_std=True,
        is_send_dingtalk=False,
        is_save_to_disk=CONFIGS.is_save_log_to_disk,
        log_path=os.path.join(LOGGING_PATH, LOGGING_FILENAME)
    )

    # 特征工程数据载入
    # *******************
    f_dir = '../cached_data/train_feats'
    f_names_list = [f_name for f_name in os.listdir(f_dir) if f_name.endswith('.csv')]

    train_feats_df_list = []
    for f_name in f_names_list:
        train_feats_df_list.append(pd.read_csv(os.path.join(f_dir, f_name)))

    train_feats_df = pd.concat(train_feats_df_list, ignore_index=True)
    train_feats_df['unix_ts'] = (pd.to_datetime(train_feats_df['unix_ts']).astype(int) / 10**9).astype(int)
    train_feats_df['label'] = train_feats_df['label'].astype(int)

    # 模型训练与交叉验证
    # *******************

    # 训练数据元信息
    # ----------
    key_feat_names = ['kpi_id', 'unix_ts']

    category_feat_names = [
        name for name in train_feats_df.columns if 'category' in name
    ]
    numeric_feat_names = [
        name for name in train_feats_df.columns if 'numeric' in name
    ]

    train_target_vals = train_feats_df['label'].values
    train_feats_df_list = [
        item[1].reset_index(drop=True) for item in list(train_feats_df.groupby(['kpi_id']))
    ]

    # 交叉验证相关参数
    # ----------
    valid_score_cols = [
        'fold_id', 'valid_custom_score', 'valid_f1', 'valid_acc'
    ]
    valid_score_df = np.zeros((CONFIGS.n_folds, len(valid_score_cols)))

    # 交叉验证
    # ----------
    logger.info('TRAINING START...')
    logger.info('==================================')
    logger.info('train shape: {}'.format(train_feats_df.shape))
    for fold, X_train, X_valid in generator_tscv(train_feats_df_list, CONFIGS.n_folds):

        # STEP 1: 依据训练与验证的bool索引, 构建训练与验证对应的数据集
        # ----------
        y_train = X_train['label'].values
        y_valid = X_valid['label'].values

        X_train_feats = X_train.drop(key_feat_names + ['label'], axis=1)
        X_valid_feats = X_valid.drop(key_feat_names + ['label'], axis=1)

        xgb_params = CONFIGS.xgb_params.copy()
        if gpus:
            xgb_params['tree_method'] = 'gpu_hist'
            xgb_params['gpu_id'] = CONFIGS.gpu_id

        logger.info(
            '-- train precent: {:.3f}%, valid precent: {:.3f}%'.format(
                100 * len(X_train_feats) / len(train_feats_df),
                100 * len(X_valid_feats) / len(train_feats_df)
            )
        )

        # STEP 2: 开始训练模型
        # ----------
        # def feval_custom_metric(true_eval_df, threshold, delay):
        #     valid_eval_df = true_eval_df.copy()
        #     def compute_adjusted_f1(y_pred, dtrain):
        #         # y_true_label = dtrain.get_label()
        #         y_pred_label = np.where(y_pred > threshold, 1, 0)

        #         valid_eval_df['label'] = y_pred_label

        #         custom_score = evaluate_df_score(true_eval_df, valid_eval_df, delay)['total_score'][0]

        #         return 'adjusted_f1', -1 * np.round(custom_score, 7)
        #     return compute_adjusted_f1

        custom_metric = feval_custom_metric(CONFIGS.threshold)

        xgb_clf = xgb.XGBClassifier(**xgb_params)
        xgb_clf.fit(
            X_train_feats, y_train,
            eval_set=[(X_valid_feats, y_valid)],
            eval_metric=custom_metric,
            early_stopping_rounds=CONFIGS.early_stopping_rounds,
            verbose=CONFIGS.verbose_rounds
        )

        # STEP 3: 预测与评估
        # ----------
        y_val_pred = xgb_clf.predict_proba(
            X_valid_feats, ntree_limit=xgb_clf.best_iteration+1
        )
        y_val_pred_label = np.where(y_val_pred[:, 1] > CONFIGS.threshold, 1, 0)

        # 存储valid的oof预测结果
        X_valid_eval_df = X_valid[key_feat_names + ['label']].copy()
        X_valid_true_df = X_valid[key_feat_names + ['label']].copy()
        X_valid_true_df['label'] = y_val_pred_label

        val_custom = evaluate_df_score(
            X_valid_true_df.rename({'unix_ts': 'timestamp'}, axis=1),
            X_valid_eval_df.rename({'unix_ts': 'timestamp'}, axis=1),
            delay=7
        )['total_score'][0]
        val_f1 = f1_score(
            y_valid.reshape(-1, 1),
            y_val_pred_label.reshape(-1, 1),
            average='macro'
        )
        val_acc = accuracy_score(
            y_valid.reshape(-1, 1),
            y_val_pred_label.reshape(-1, 1)
        )

        logger.info(
            '-- fold {}({}): val custom: {:.4f}, f1: {:.4f}, acc: {:.4f}, best iters: {}\n'.format(
                fold+1, CONFIGS.n_folds, val_custom, val_f1, val_acc, xgb_clf.best_iteration
            )
        )

        valid_score_df[fold, 0] = fold
        valid_score_df[fold, 1] = val_custom
        valid_score_df[fold, 2] = val_f1
        valid_score_df[fold, 3] = val_acc

        # 保存序列化的模型
        if CONFIGS.is_save_model_to_disk:
            serialize_xgboost_model(
                xgb_clf,
                model_path='../cached_models/xgb_gpu_models', model_version=str(fold+1)
            )

    # 整体Out of fold训练指标估计
    # *******************
    valid_score_df = pd.DataFrame(
        valid_score_df, columns=valid_score_cols
    )
    valid_score_df['fold_id'] = valid_score_df['fold_id'].astype(int)

    valid_score_cols = [
        'fold_id', 'valid_custom_score', 'valid_f1', 'valid_acc'
    ]
    logger.info(
        '-- TOTAL AVG: val custom: {:.4f}, f1: {:.4f}, acc: {:.4f}'.format(
            valid_score_df['valid_custom_score'].mean(),
            valid_score_df['valid_f1'].mean(),
            valid_score_df['valid_acc'].mean()
        )
    )
    logger.info('\n' + str(valid_score_df))

    logger.info('TRAINING END...')
    logger.info('==================================')

    logger.info('\n***************FINISHED...***************')

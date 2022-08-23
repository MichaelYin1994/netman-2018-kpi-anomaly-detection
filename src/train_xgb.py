# -*- coding: utf-8 -*-

# Created on 202203072137
# Author:    zhuoyin94 <zhuoyin94@163.com>
# Github:    https://github.com/MichaelYin1994

'''
训练XGBoost分类器，使用测试集来训练分类器，在训练集全集上验证。
'''

import argparse
import multiprocessing as mp
import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

from metrics.metrics import evaluate_df_score
from utils.io_utils import LoadSave, serialize_xgboost_model
from utils.logger import get_datetime, get_logger

gpus = tf.config.experimental.list_physical_devices('GPU')


# def custom_score_metric(y_pred, dtrain):
#     '''官方Metrics的XGBoost实现，Higher better'''
#     y_true_label = dtrain.get_label()
#     y_pred_label = np.argmax(y_pred, axis=1)
#     score = compute_custom_score(y_true_label, y_pred_label)

#     return 'custom_score', -1 * score


class CONFIGS:
    n_folds = 5
    task_name = 'train_xgb'
    cv_strategy = 'gkf'
    early_stopping_rounds = 2000

    gpu_id = 0
    verbose_rounds = 300
    global_random_seed = 2077
    is_save_log_to_disk = False
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
        'subsample': 0.95,
        'random_state': global_random_seed
    }


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
    # train_feats_df = pd.read_csv('../cached_data/train_feats_df.csv')

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
    train_feats_df.drop(['label'], axis=1, inplace=True)

    # 编码category变量
    # ----------
    for feat_name in category_feat_names:
        train_feats_df[feat_name] = train_feats_df[feat_name].astype('category')
        train_feats_df[feat_name] = train_feats_df[feat_name].astype('category')

    # 交叉验证相关参数
    # ----------
    valid_score_cols = [
        'fold_id', 'valid_custom_score', 'valid_f1', 'valid_acc'
    ]
    valid_score_df = np.zeros((CONFIGS.n_folds, len(valid_score_cols)))
    test_pred_df_list = []
    oof_pred_proba_df = np.zeros((len(train_feats_df), 2))

    if CONFIGS.cv_strategy == 'kf':
        folds = KFold(n_splits=CONFIGS.n_folds)
        fold_generator = folds.split(
            np.arange(0, len(train_feats_df)),
        )
    elif CONFIGS.cv_strategy == 'gkf':
        folds = GroupKFold(n_splits=CONFIGS.n_folds)
        fold_generator = folds.split(
            np.arange(0, len(train_feats_df)), None,
            groups=train_feats_df['kpi_id'].values
        )
    elif CONFIGS.cv_strategy == 'skf':
        folds = StratifiedKFold(
            n_splits=CONFIGS.n_folds, random_state=CONFIGS.global_random_seed, shuffle=True
        )
        fold_generator = folds.split(
            np.arange(0, len(train_feats_df)), train_target_vals
        )

    logger.info(
        'Cross validation strategy: {}'.format(CONFIGS.cv_strategy)
    )

    # 交叉验证
    # ----------
    logger.info('TRAINING START...')
    logger.info('==================================')
    logger.info('train shape: {}, test shape: {}'.format(
        train_feats_df.shape, train_feats_df.shape
    ))
    for fold, (train_idx, valid_idx) in enumerate(fold_generator):

        # STEP 1: 依据训练与验证的bool索引, 构建训练与验证对应的数据集
        # ----------
        X_train = train_feats_df.iloc[train_idx].drop(key_feat_names, axis=1)
        X_valid = train_feats_df.iloc[valid_idx].drop(key_feat_names, axis=1)

        y_train = train_target_vals[train_idx]
        y_valid = train_target_vals[valid_idx]

        xgb_params = CONFIGS.xgb_params.copy()

        if gpus:
            xgb_params['tree_method'] = 'gpu_hist'
            xgb_params['gpu_id'] = CONFIGS.gpu_id

        logger.info(
            '-- train precent: {:.3f}%, valid precent: {:.3f}%'.format(
                100 * len(X_train) / len(train_feats_df),
                100 * len(X_valid) / len(train_feats_df)
            )
        )

        # STEP 2: 开始训练模型
        # ----------
        xgb_clf = xgb.XGBClassifier(**xgb_params)
        xgb_clf.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric='auc',
            early_stopping_rounds=CONFIGS.early_stopping_rounds,
            verbose=CONFIGS.verbose_rounds
        )

        # STEP 3: 预测与评估
        # ----------
        y_val_pred = xgb_clf.predict_proba(
            X_valid, ntree_limit=xgb_clf.best_iteration+1
        )
        y_val_pred_label = np.argmax(y_val_pred, axis=1)

        # 存储valid的oof预测结果
        oof_pred_proba_df[valid_idx, :] = y_val_pred

        X_valid_eval_df = train_feats_df.iloc[valid_idx][key_feat_names]
        X_valid_eval_df['label'] = y_val_pred_label
        X_valid_true_df = train_feats_df.iloc[valid_idx][key_feat_names]
        X_valid_true_df['label'] = y_valid

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

    train_oof_pred_vals = np.argmax(oof_pred_proba_df, axis=-1)
    total_eval_df = train_feats_df[key_feat_names].rename({'unix_ts': 'timestamp'}, axis=1)
    total_eval_df['label'] = train_target_vals
    total_pred_df = total_eval_df.copy()
    total_pred_df['label'] = train_oof_pred_vals

    global_custom = evaluate_df_score(
        total_eval_df, total_pred_df, delay=7
    )['total_score'][0]
    global_f1 = f1_score(
        train_target_vals.reshape(-1, 1),
        train_oof_pred_vals.reshape(-1, 1),
        average='macro'
    )
    global_acc = accuracy_score(
        train_target_vals.reshape(-1, 1),
        train_oof_pred_vals.reshape(-1, 1)
    )

    logger.info(
        '-- TOTAL OOF: val custom: {:.4f}, f1: {:.4f}, acc: {:.4f}'.format(
            global_custom, global_f1, global_acc
        )
    )
    logger.info(
        '\n' + str(classification_report(
            train_target_vals, train_oof_pred_vals, digits=4)
        )
    )
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

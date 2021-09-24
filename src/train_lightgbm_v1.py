#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202109232328
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
本模块（train_lightgbm_v2.py）采用定阈值法训练lgb分类器。
'''
import gc
import os
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from utils import LoadSave, evaluate_df_score, pr_auc_score, njit_f1, adjust_predict_label

GLOBAL_RANDOM_SEED = 2021
###############################################################################

def generator_tscv(df_list, n_folds=2, is_shuffle_train=True):
    '''Generator of the data'''
    n_splits = n_folds + 1
    df_size_list = [len(item) for item in df_list]

    for i in range(n_folds):
        train_idx_list = [
            int(np.floor(item / n_splits * (i + 1))) for item in df_size_list
        ]
        valid_idx_list = [
            int(np.floor(item / n_splits * (i + 2))) for item in df_size_list
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
                frac=1, random_state=GLOBAL_RANDOM_SEED
            )
            df_train.reset_index(drop=True, inplace=True)

        yield i, df_train, df_valid


def feval_custom_metric(threshold):

    def compute_custom_f1_score(y_true, y_pred):
        y_pred_label = np.where(y_pred > threshold, 1, 0)
        y_pred_label_adjusted = adjust_predict_label(
            y_true, y_pred_label, delay=7
        )

        custom_score, _, _ = njit_f1(y_true, y_pred_label_adjusted)

        return 'adjusted_f1', np.round(custom_score, 7), True

    return compute_custom_f1_score


if __name__ == '__main__':
    # 载入特征数据
    # ----------------
    N_FOLDS = 5
    N_SEARCH = 12
    THRESHOLD_LOW, THRESHOLD_HIGH = 0.35, 0.6
    VERBOSE = 0
    EARLY_STOPPING_ROUNDS = 500

    lgb_params = {'boosting_type': 'gbdt',
                  'objective': 'binary',
                  'metric': 'custom',
                  'n_estimators': 10000,
                  'num_leaves': 31,
                  'max_depth': 4,
                  'learning_rate': 0.09,
                  'colsample_bytree': 0.95,
                  'subsample': 0.95,
                  'subsample_freq': 1,
                  'reg_alpha': 0,
                  'reg_lambda': 0.01,
                  'random_state': GLOBAL_RANDOM_SEED,
                  'n_jobs': -1
                  }

    # 载入特征数据
    # ----------------
    file_processor = LoadSave(dir_name='../cached_data/')
    train_feats_list = file_processor.load_data(
        file_name='train_feats_list.pkl'
    )

    # STAGE 1: 定阈值训练LightGBM分类器，优化官方Metric
    # ----------------
    threshold_list = np.linspace(
        THRESHOLD_LOW, THRESHOLD_HIGH, N_SEARCH
    ).tolist()
    threshold_list += [0.5]
    threshold_list = list(set(threshold_list))
    threshold_list = sorted(threshold_list)

    print('\n[INFO] {} Search for the best F1...'.format(
        str(datetime.now())[:-4]))
    print('**********************')
    print('[INFO] train shape: {}, total folds: {}'.format(
        (np.sum([len(df) for df in train_feats_list]),
         train_feats_list[0].shape[1]),
        N_FOLDS)
    )

    best_f1, best_threshold, best_model_name = -1, 0.5, None
    best_model_list = None

    for idx, decision_threshold in enumerate(threshold_list):

        print(
            '\n[INFO] {} Current threshold: {:.8f}({}/{})'.format(
                str(datetime.now())[:-4], decision_threshold, idx+1, len(threshold_list)
            )
        )
        print('+++++++++++')
        y_val_score_df = np.zeros((N_FOLDS, 4))
        trained_model_list, valid_df_list = [], []
        for fold, train_df, val_df in generator_tscv(train_feats_list, N_FOLDS):
            train_feats = train_df.drop(['label', 'timestamp'], axis=1).values
            val_feats = val_df.drop(['label', 'timestamp'], axis=1).values

            train_label = train_df['label'].values
            val_label = val_df['label'].values

            # 训练lightgbm模型
            compute_custom_f1 = feval_custom_metric(decision_threshold)

            clf = lgb.LGBMClassifier(**lgb_params)
            clf.fit(
                train_feats, train_label,
                eval_set=[(val_feats, val_label)],
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                eval_metric=compute_custom_f1,
                categorical_feature=[0],
                verbose=VERBOSE
            )
            trained_model_list.append(clf)

            # 对validation数据进行预测
            val_pred_df = val_df[['kpi_id', 'label', 'timestamp']].copy()
            val_pred_df.rename({'label': 'true_label'}, axis=1, inplace=True)

            val_pred_df['label'] = np.where(
                clf.predict_proba(
                    val_feats, num_iteration=clf.best_iteration_
                )[:, 1] > decision_threshold, 1, 0
            )
            valid_df_list.append(val_pred_df)

            # 评估效果
            y_val_score_df[fold, 0] = fold
            y_val_score_df[fold, 1] = clf.best_iteration_
            y_val_score_df[fold, 2] = np.mean(
                evaluate_df_score(
                    val_df[['kpi_id', 'label', 'timestamp']],
                    val_pred_df, delay=7
                )['f1_score_list']
            )
            y_val_score_df[fold, 3] = decision_threshold

            print(
                '-- {} folds {}({}), valid iters: {}, eval_f1 {:7f}'.format(
                    str(datetime.now())[:-4], fold+1, N_FOLDS,
                    int(y_val_score_df[fold, 1]),
                    y_val_score_df[fold, 2]
                )
            )

        # 计算整体预测效果
        val_total_df = pd.concat(valid_df_list)
        val_pred_df = val_total_df[['kpi_id', 'label', 'timestamp']]
        val_true_df = val_total_df[['kpi_id', 'true_label', 'timestamp']]
        val_true_df.rename({'true_label': 'label'}, axis=1, inplace=True)

        val_total_f1 = np.mean(
            evaluate_df_score(
                val_true_df, val_pred_df, delay=7
            )['f1_score_list']
        )
        y_val_score_df = pd.DataFrame(
            y_val_score_df,
            columns=['folds', 'best_iter', 'val_adjusted_f1', 'val_best_threshold']
        )

        # 保存CV的日志信息
        y_val_score_df = pd.DataFrame(
            y_val_score_df,
            columns=['folds', 'best_iter', 'val_adjusted_f1', 'val_best_threshold']
        )

        sub_file_name = '{}_lgb_nfolds_{}_valf1_{}_totalf1_{}'.format(
            len(os.listdir('../logs/')) + 1,
            N_FOLDS,
            str(np.round(y_val_score_df['val_adjusted_f1'].mean(), 6)).split('.')[1],
            str(np.round(val_total_f1, 6)).split('.')[1],
        )
        y_val_score_df.to_csv('../logs/{}.csv'.format(sub_file_name), index=False)

        print(
            '-- {} TOTAL valid f1: {:.7f}'.format(
                str(datetime.now())[:-4], val_total_f1
            )
        )
        print('+++++++++++')

        # 保存CV的日志信息
        if val_total_f1 > best_f1:
            best_f1 = val_total_f1
            best_model_list = trained_model_list
            best_threshold = decision_threshold
            best_model_name = sub_file_name

    print('\n**********************')
    print('[INFO] {} LightGBM training end...\n'.format(
        str(datetime.now())[:-4]))

    # STAGE 2: 保存训练好的模型/阈值与训练日志
    # ----------------
    file_processor = LoadSave(dir_name='../models/')
    file_processor.save_data(
        file_name='{}.pkl'.format(best_model_name),
        data_file=[best_model_list, best_threshold]
    )

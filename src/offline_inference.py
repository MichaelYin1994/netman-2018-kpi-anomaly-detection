#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202109021528
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
本模块（offline_inference.py）离线load测试数据，构造离线特征，
并对测试数据进行inference和结果评估。
'''

import gc
import warnings
from datetime import datetime
import multiprocessing as mp

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import LoadSave, evaluate_df_score
from compute_feature_engineering import compute_kpi_feats_df

# 设定全局随机种子，并且屏蔽warnings
GLOBAL_RANDOM_SEED = 2021
np.random.seed(GLOBAL_RANDOM_SEED)
warnings.filterwarnings('ignore')
###############################################################################
def fcn(params):
    total_feats = compute_kpi_feats_df(params[0], params[1])
    return total_feats


def evaluate_test_lgb(model_name, file_name):
    '''指定测试集上评估lgb模型的效果'''
    # 载入test_df数据
    # ----------------
    file_processor = LoadSave(dir_name='../cached_data/')
    test_df = file_processor.load_data(file_name=file_name)

    # 载入trained models，并进行编译加速
    # https://treelite.readthedocs.io/en/latest/
    # ----------------
    file_processor = LoadSave(dir_name='../models/')
    trained_models_list, decision_threshold = file_processor.load_data(
        file_name=MODEL_FILE_NAME
    )

    # 载入训练数据的StreamDeque对象
    # ----------------
    file_processor = LoadSave(dir_name='../cached_data/')
    stream_deque_dict = file_processor.load_data(
        file_name='train_stream_deque_dict.pkl'
    )

    # 按kpi_id拆分训练数据
    # ----------------
    unique_kpi_ids = test_df['kpi_id'].unique()

    test_df_list = []
    for kpi_id in unique_kpi_ids:
        tmp_df = test_df[test_df['kpi_id'] == kpi_id].reset_index(drop=True)
        test_df_list.append([tmp_df, stream_deque_dict[kpi_id]])

    del test_df
    gc.collect()

    # 多进程并行特征工程
    # ----------------
    with mp.Pool(processes=mp.cpu_count(), maxtasksperchild=1) as p:
        feats_obj_list = list(tqdm(p.imap(
            fcn, test_df_list
        ), total=len(test_df_list)))

    # 获取分模型预测结果
    # ----------------
    stream_deque_dict = {}
    for i, item in enumerate(feats_obj_list):
        stream_deque_dict[i] = item[1]
    test_feats_df = pd.concat(
        [item[0] for item in feats_obj_list], axis=0, ignore_index=True
    )

    test_feats = test_feats_df.drop(['label', 'timestamp'], axis=1).values

    print('\n[INFO] {} Offline testing evaluation...'.format(
        str(datetime.now())[:-4]))
    print('==================================')
    test_pred_proba_list = []
    for fold, model in enumerate(trained_models_list):
        test_pred_proba_list.append(model.predict_proba(
            test_feats, num_iteration=model.best_iteration_
        )[:, 1])

        # 评估效果
        test_pred_df = test_feats_df[['kpi_id', 'label', 'timestamp']].copy()
        test_pred_df['label'] = np.where(
            test_pred_proba_list[-1] >= decision_threshold, 1, 0
        )
        test_score_dict = evaluate_df_score(test_feats_df, test_pred_df)

        print('-- {} MEAN f1: {:.4f}, precision: {:.4f}, recall: {:.4f}, TOTAL f1: {:.4f}, precision: {:.4f}, recall: {:.4f}'.format(
            str(datetime.now())[:-4], np.mean(test_score_dict['f1_score_list']),
            np.mean(test_score_dict['precision_score_list']),
            np.mean(test_score_dict['recall_score_list']),
            test_score_dict['total_score'][0],
            test_score_dict['total_score'][1],
            test_score_dict['total_score'][2]
        ))

    # 评估效果
    test_pred_proba = np.mean(test_pred_proba_list, axis=0)
    test_pred_df = test_feats_df[['kpi_id', 'label', 'timestamp']].copy()
    test_pred_df['label'] = np.where(
        test_pred_proba >= decision_threshold, 1, 0
    )
    test_score_dict = evaluate_df_score(test_feats_df, test_pred_df)

    print('-- {} FINAL MEAN f1: {:.4f}, precision: {:.4f}, recall: {:.4f}, TOTAL f1: {:.4f}, precision: {:.4f}, recall: {:.4f}'.format(
        str(datetime.now())[:-4], np.mean(test_score_dict['f1_score_list']),
        np.mean(test_score_dict['precision_score_list']),
        np.mean(test_score_dict['recall_score_list']),
        test_score_dict['total_score'][0],
        test_score_dict['total_score'][1],
        test_score_dict['total_score'][2]
    ))

    print('==================================')


if __name__ == '__main__':
    MODEL_FILE_NAME = '53_lgb_nfolds_5_valf1_514573_totalf1_531719.pkl'
    TEST_FILE_NAME = 'test_df_part_x.pkl'  # test_df_part_x, test_df_part_y, test_df

    # 载入test_df数据
    # ----------------
    evaluate_test_lgb(MODEL_FILE_NAME, TEST_FILE_NAME)
    
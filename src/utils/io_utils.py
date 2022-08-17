# -*- coding: utf-8 -*-

# Created on 202108311136
# Author:    zhuoyin94 <zhuoyin94@163.com>
# Github:    https://github.com/MichaelYin1994

'''
数据处理与特征工程辅助代码。
'''
import os
import pickle
import re
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from easydict import EasyDict as edict
from tqdm import tqdm

warnings.filterwarnings('ignore')
###############################################################################

def load_from_csv(dir_name, file_name, **kwargs):
    '''从dir_name路径读取file_name的文件'''
    if not isinstance(dir_name, str) or not isinstance(file_name, str):
        raise TypeError('Input dir_name and file_name must be str type !')
    if not file_name.endswith('.csv'):
        raise ValueError('Input file_name must end with *.csv !')

    full_name = os.path.join(dir_name, file_name)
    return pd.read_csv(full_name, **kwargs)


class LoadSave():
    '''以*.pkl格式，利用pickle包存储各种形式（*.npz, list etc.）的数据。

    Attributes:
    ----------
        dir_name: {str-like}
            数据希望读取/存储的路径信息。
        file_name: {str-like}
            希望读取与存储的数据文件名。
        verbose: {int-like}
            是否打印存储路径信息。
    '''
    def __init__(self, dir_name=None, file_name=None, verbose=1):
        if dir_name is None:
            self.dir_name = './data_tmp/'
        else:
            self.dir_name = dir_name
        self.file_name = file_name
        self.verbose = verbose

    def save_data(self, dir_name=None, file_name=None, data_file=None):
        '''将data_file保存到dir_name下以file_name命名。'''
        if data_file is None:
            raise ValueError('LoadSave: Empty data_file !')

        if dir_name is None or not isinstance(dir_name, str):
            dir_name = self.dir_name
        if file_name is None:
            file_name = self.file_name
        if not isinstance(file_name, str) or not file_name.endswith('.pkl'):
            raise ValueError('LoadSave: Invalid file_name !')

        # 保存数据以指定名称到指定路径
        full_name = os.path.join(
            dir_name, file_name
        )
        with open(full_name, 'wb') as file_obj:
            pickle.dump(data_file, file_obj, protocol=4)

        if self.verbose:
            print('[INFO] {} LoadSave: Save to dir {} with name {}'.format(
                str(datetime.now())[:-7], dir_name, file_name)
            )

    def load_data(self, dir_name=None, file_name=None):
        '''从指定的dir_name载入名字为file_name的文件到内存里。'''
        if dir_name is None or not isinstance(dir_name, str):
            dir_name = self.dir_name
        if file_name is None:
            file_name = self.file_name
        if not isinstance(file_name, str) or not file_name.endswith('.pkl'):
            raise ValueError('LoadSave: Invalid file_name !')

        # 从指定路径导入指定文件名的数据
        full_name = os.path.join(
            dir_name, file_name
        )
        with open(full_name, 'rb') as file_obj:
            data_loaded = pickle.load(file_obj)

        if self.verbose:
            print('[INFO] {} LoadSave: Load from dir {} with name {}'.format(
                str(datetime.now())[:-7], dir_name, file_name)
            )
        return data_loaded

# https://gist.github.com/wphicks/2298fe904f3293a59254e0c35cfe05c1#file-fraud_detection_example-ipynb

def serialize_xgboost_model(xgb_model, model_path, model_version='1'):
    if model_path == None:
        os.makedirs('./cached_model', exist_ok=True)
        model_path = './cached_model'

    # 构建model的文件夹
    version_dir = os.path.join(model_path, str(model_version))
    os.makedirs(version_dir, exist_ok=True)

    # 存储xgboost模型
    xgb_model.save_model(
        os.path.join(version_dir, 'xgboost.json')
    )


class YamlParser:
    '''yaml文件读取、解析与存储'''
    def __init__(self, config_dict=None, config_file=None):
        self.config_dict = edict(config_dict)

        if not (config_file is None):
            assert os.path.isfile(config_file), 'The config file {config_file} does not exist!'

            with open(config_file, 'r') as f:
                cfg = yaml.safe_load(f)
                self.config_dict.update(cfg)

    def merge_from_file(self, config_file):
        with open(config_file, 'r') as f:
            cfg = yaml.safe_load(f)
            self.config_dict.update(cfg)

    def merge_from_dict(self, config_dict):
        self.config_dict.update(config_dict)

    def edict2dict(self, obj):
        if type(obj) not in [list, dict]:
            return

        if type(obj) in list:
            for i in range(len(obj)):
                if isinstance(obj[i], edict):
                    obj[i] = dict(obj[i])
                self.edict2dict(obj[i])
        else:
            for k, v in obj.items():
                if isinstance(v, edict):
                    obj[k] = dict(obj[k])
                self.edict2dict(obj[k])
        return

    def save(self, dir, f_name):
        # Transform edict object to dict
        config_dict = dict(self.config_dict.copy())
        self.edict2dict(config_dict)

        # save to the local dir
        full_name = os.path.join(dir, f_name)
        with open(full_name, 'w', encoding='utf-8') as f:
            yaml.dump(
                data=config_dict, stream=f, allow_unicode=True
            )
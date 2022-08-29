# -*- coding: utf-8 -*-

# Created on 202208291430
# Author:    zhuoyin94 <zhuoyin94@163.com>
# Github:    https://github.com/MichaelYin1994

'''
测试在线特征工程API。
'''

import json
import os
import sys

sys.path.append('..')

import requests
from tqdm import tqdm

from utils.configs import fe_server_configs as CONFIGS
from utils.io_utils import LoadSave

# 测试数据载入
# *******************
file_handler = LoadSave(dir_name='../../cached_data/')
test_df = file_handler.load_data(file_name='test_df_part_x.pkl')

test_df = test_df.iloc[:10].reset_index(drop=True)

# 将数据类型进行转换
test_df_dict = test_df.to_dict(orient='list')
col_name_list = list(test_df.columns)

total_size, res_list = len(test_df), []
for i in tqdm(range(0, len(test_df))):
    data = {'input': []}

    # 构建样本
    sample = []
    for col_name in col_name_list:
        sample.append(test_df_dict[col_name][i])
    data['input'].append(sample)

    # 发送样本获取结果
    response = requests.post(
        CONFIGS.url, data=json.dumps(data), headers=CONFIGS.headers
    ).json()

    # 保存推理结果
    res_list.append(response)

print(res_list)

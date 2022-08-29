# -*- coding: utf-8 -*-

# Created on 202208261446
# Author:    zhuoyin94 <zhuoyin94@163.com>
# Github:    https://github.com/MichaelYin1994

'''
用于Triton Inference Server模型部署的工具。
'''

import json
import os
import time

import tritonclient.grpc as triton_grpc
from tritonclient import utils as triton_utils


def check_triton_server_status(triton_client, timeout=30):
    '''检查Triton Inference Server的服务器状态'''

    start, count = time.time(), 0
    while True:
        try:
            if triton_client.is_server_ready() or time.time() - start > timeout:
                print('Server is ready? {} !'.format(triton_client.is_server_ready()))
                break
        except triton_utils.InferenceServerException:
            print('TRITON INFERENCE SERVER IS NOT READY ! CHECK {} TIMES !'.format(count))
            count += 1

            if count > timeout:
                raise triton_utils.InferenceServerException('Invalid Triton Server !')
        time.sleep(1)


def get_grpc_triton_predict_proba(client, model_name, model_version, feats_mat):
    '''调用client发送请求到triton inference server，获取预测结果'''

    # 组装input数据
    triton_input = triton_grpc.InferInput(
        name='input__0', shape=feats_mat.shape, datatype='FP32'
    )
    triton_input.set_data_from_numpy(feats_mat)

    # output数据
    triton_output = triton_grpc.InferRequestedOutput('output__0')
    response = client.infer(
        model_name, model_version=model_version, inputs=[triton_input], outputs=[triton_output]
    )

    return response.as_numpy('output__0')


def generate_xgboost_deployment_config(
    deployment_type='gpu',
    storage_type='AUTO',
    n_features=None,
    n_classes=None,
    instance_count=2,
    max_memory_bytes=67108864
    ):
    '''生成Triton inference server模型部署的配置文件'''

    # 模型部署方式
    if deployment_type.lower() == 'cpu':
        instance_kind = 'KIND_CPU'
    else:
        instance_kind = 'KIND_GPU'

    # 推理参数（按照fp32计算）
    bytes_per_sample = (n_features + n_classes) * 4
    max_batch_size = max_memory_bytes // bytes_per_sample

    # 部署配置文件生成
    config_text = (
        # 后端信息
        f'backend: "fil"',
        f'max_batch_size: {max_batch_size}',
        # 输入配置信息
        f"""input [
            {{
                name: "input__0"
                data_type: TYPE_FP32
                dims: [{n_features}]
            }}
        ]""",
        # 输出配置信息
        f"""output [
            {{
                name: "output__0"
                data_type: TYPE_FP32
                dims: [{n_classes}]
            }}
        ]""",
        # 多模型推理配置
        f"""instance_group [
            {{
                count: {instance_count}
                kind: {instance_kind}
                gpus: [0]
            }}
        ]""",
        # XGBoost模型参数配置信息
        f"""parameters [
        {{
            key: "model_type"
            value: {{ string_value: "xgboost_json" }}
        }},
        {{
            key: "predict_proba"
            value: {{ string_value: "true" }}
        }},
        {{
            key: "output_class"
            value: {{ string_value: "true" }}
        }},
        {{
            key: "threshold"
            value: {{ string_value: "0.5" }}
        }},
        {{
            key: "storage_type"
            value: {{ string_value: "{storage_type}" }}
        }}
        ]
        """,
        # batching策略
        f"""dynamic_batching
        {{
            max_queue_delay_microseconds: 100
        }}
        """
        # 版本策略（默认部署model repository）全部模型
        f"""version_policy:
        {{
            all {{ }}
        }}
        """
    )

    return config_text
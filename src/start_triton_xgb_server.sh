# -*- coding: utf-8 -*-

# Created on 202208261020
# Author:    zhuoyin94 <zhuoyin94@163.com>
# Github:    https://github.com/MichaelYin1994

# 需要预先生成模型部署的configs.pbtxt文件！模型部署命令：
# docker run --gpus all -d -p 8000:8000 -p 8001:8001 -p 8002:8002 -v {REPO_PATH}:/models --name tritonserver {TRITON_IMAGE} tritonserver --model-repository=/models

export TRITON_IMAGE_NAME="serving/triton-server:22.05-py3"
export GPU_ID="0"
export XGBOOST_MODEL_PATH="/home/zhuoyin94/Desktop/hard_disk/kpi-anomaly-detection/cached_models/"
export INFERENCE_SERVER_NAME="serving_tritonserver_xgb"

docker run --gpus all --rm -d -p 8000:8000 -p 8001:8001 -p 8002:8002 -v /etc/localtime:/etc/localtime:ro -v ${XGBOOST_MODEL_PATH}:/models --name ${INFERENCE_SERVER_NAME} ${TRITON_IMAGE_NAME} tritonserver --model-repository=/models

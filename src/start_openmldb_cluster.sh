# -*- coding: utf-8 -*-

# Created on 202208261022
# Author:    zhuoyin94 <zhuoyin94@163.com>
# Github:    https://github.com/MichaelYin1994

# 集群版CLI启动(tct-cvpr)：./openmldb/bin/openmldb --zk_cluster=127.0.0.1:2181 --zk_root_path=/openmldb --role=sql_client

export PROJECT_PATH="/home/zhuoyin94/Desktop/hard_disk/kpi-anomaly-detection/"
export CONTAINER_WORK_PATH="/work/kpi-anomaly-detection/"

docker run -it --rm -v $PROJECT_PATH:$CONTAINER_WORK_PATH -v /etc/localtime:/etc/localtime:ro --network=host --name servering_openmldb mlops/openmldb:0.6.0

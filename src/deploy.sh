# cluster版本：-p 9080:9080 -p 7527:7527 -p 10921:10921 -p 10922:10922
# standalone版本：-p 6527:6527 -p 9921:9921 -p 8080:8080
# 全家桶版本：-p 9080:9080 -p 7527:7527 -p 10921:10921 -p 10922:10922 -p 6527:6527 -p 9921:9921 -p 8080:8080

# 集群版CLI启动(tct-cvpr)：./openmldb/bin/openmldb --zk_cluster=127.0.0.1:2181 --zk_root_path=/openmldb --role=sql_client

export PROJECT_PATH="/home/tct-cvpr/Desktop/yinzhuo_host_files/kpi-anomaly-detection/"
export CONTAINER_WORK_PATH="/work/kpi-anomaly-detection/"
docker run -it --rm -v $PROJECT_PATH:$CONTAINER_WORK_PATH -v /etc/localtime:/etc/localtime:ro --network=host --name servering_openmldb mlops/openmldb:0.6.0

# 映射项目地址
# ./openmldb/bin/openmldb --host 127.0.0.1 --port 6527

# cluster版本：-p 9080:9080 -p 7527:7527 -p 10921:10921 -p 10922:10922
# standalone版本：-p 6527:6527 -p 9921:9921 -p 8080:8080
# 全家桶版本：-p 9080:9080 -p 7527:7527 -p 10921:10921 -p 10922:10922 -p 6527:6527 -p 9921:9921 -p 8080:8080

# ./openmldb/bin/openmldb --host 0.0.0.0 --port 6527

export PROJECT_PATH="/home/zhuoyin94/Desktop/hard_disk/kpi-anomaly-detection"
export CONTAINER_WORK_PATH="/work/kpi-anomaly-detection"
docker run -it --rm -v $PROJECT_PATH:$CONTAINER_WORK_PATH -v /etc/localtime:/etc/localtime:ro --network=host --name servering_openmldb mlops/openmldb:0.6.0

## 2018年AIOps国际挑战赛KPI异常检测工程化部署方法实践

---
### **作者简介**

作者：鱼丸粗面（zhuoyin94@163.com）。整体采用了[此项目](https://github.com/MichaelYin1994/python-style-guide)的编码规范，逐步完善中（20220617）。

---
### **系统环境与组件依赖**

系统环境:
- Ubuntu 20.04 LTS
- GPU: NVIDIA Corporation GP104GL [Quadro P5000](16G)
- CPU: Intel® Core™ i9-9920X CPU @ 3.50GHz × 24
- RAM: 94G
- CUDA: 11.4
- swap: 64G

使用开源组件:
- OpenMLDB: 用于stream形式数据的实时特征工程。
- cAdivisor: 用于容器状态监控。
- Triton inference server: 用于serving XGBoost模型与DL-based模型。
- Prometheus + Grafana: 用于容器状态的可视化监控。

---
### **References**

[1] https://github.com/MichaelYin1994/tianchi-pakdd-aiops-2021

[2] Lam, Siu Kwan, Antoine Pitrou, and Stanley Seibert. "Numba: A llvm-based python jit compiler." Proceedings of the Second Workshop on the LLVM Compiler Infrastructure in HPC. 2015.

[3] https://github.com/johannfaouzi/pyts

[4] https://github.com/blue-yonder/tsfresh

[5] Goldstein M, Dengel A. Histogram-based outlier score (hbos): A fast unsupervised anomaly detection algorithm[J]. KI-2012: Poster and Demo Track, 2012: 59-63.

[6] Bu, Jiahao, et al. "Rapid deployment of anomaly detection models for large number of emerging kpi streams." 2018 IEEE 37th International Performance Computing and Communications Conference (IPCCC). IEEE, 2018.

[7] Ma M, Zhang S, Pei D, et al. Robust and rapid adaption for concept drift in software system anomaly detection[C]//2018 IEEE 29th International Symposium on Software Reliability Engineering (ISSRE). IEEE, 2018: 13-24.

[8] Li, Zhihan, et al. "Robust and rapid clustering of kpis for large-scale anomaly detection." 2018 IEEE/ACM 26th International Symposium on Quality of Service (IWQoS). IEEE, 2018.

[9] Li, Zeyan, Wenxiao Chen, and Dan Pei. "Robust and unsupervised kpi anomaly detection based on conditional variational autoencoder." 2018 IEEE 37th International Performance Computing and Communications Conference (IPCCC). IEEE, 2018.

[10] Liu, Dapeng, et al. "Opprentice: Towards practical and automatic anomaly detection through machine learning." Proceedings of the 2015 Internet Measurement Conference. 2015.

[11] Zhao, Nengwen, et al. "Label-less: A semi-automatic labelling tool for kpi anomalies." IEEE INFOCOM 2019-IEEE Conference on Computer Communications. IEEE, 2019.


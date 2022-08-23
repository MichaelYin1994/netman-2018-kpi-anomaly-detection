SELECT kpi_id, unix_ts, label, value FROM kpi_history_series
INTO OUTFILE '/work/kpi-anomaly-detection/cached_data/train_feats' OPTIONS(mode='overwrite');
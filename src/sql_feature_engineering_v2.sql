SELECT kpi_id, unix_ts, label, value FROM kpi_history_series
INTO OUTFILE '../cached_data/train_feats_df.csv';

SELECT id, unix_ts, label, value FROM ts_db
INTO OUTFILE '../cached_data/tmp.csv';

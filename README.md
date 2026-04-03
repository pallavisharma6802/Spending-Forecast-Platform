# Fintech Spending Analyzer

A big data pipeline for personal finance analysis and budget recommendation.

## Tech Stack

- HDFS - raw transaction storage
- Apache Hive - SQL warehouse layer
- MapReduce - batch aggregations via Hive/YARN
- Apache Spark - feature engineering and ML pipeline

## Setup

1. Start the cluster: `docker compose up -d`
2. Copy data to namenode: `docker cp spending_patterns_detailed.csv fintech-spending-analyzer-namenode-1:/tmp/`
3. Load to HDFS: `docker exec fintech-spending-analyzer-namenode-1 bash /tmp/01_load_to_hdfs.sh`
4. Create Hive tables: run `scripts/02_create_hive_tables.sql` in beeline
5. Run feature engineering: `scripts/03_feature_engineering.py`

## Pipeline

Raw CSV -> HDFS -> Hive tables -> Spark feature engineering -> Baseline saved to HDFS

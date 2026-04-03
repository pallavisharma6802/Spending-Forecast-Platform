CREATE DATABASE IF NOT EXISTS fintech
LOCATION 'hdfs://namenode:8020/user/hive/warehouse/fintech.db';

USE fintech;

CREATE EXTERNAL TABLE IF NOT EXISTS transactions (
  customer_id STRING,
  category STRING,
  item STRING,
  quantity INT,
  price_per_unit DOUBLE,
  total_spent DOUBLE,
  payment_method STRING,
  location STRING,
  transaction_date DATE
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION 'hdfs://namenode:8020/user/fintech/transactions/'
TBLPROPERTIES ("skip.header.line.count"="1");

CREATE EXTERNAL TABLE IF NOT EXISTS customer_baseline (
  customer_id STRING,
  category STRING,
  total_spend DOUBLE,
  avg_per_transaction DOUBLE,
  num_transactions INT,
  max_30d_spend DOUBLE
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION 'hdfs://namenode:8020/user/fintech/baseline/'
TBLPROPERTIES ("skip.header.line.count"="1");

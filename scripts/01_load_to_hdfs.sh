#!/bin/bash
echo "Creating HDFS directories..."
hdfs dfs -mkdir -p /user/fintech/transactions
hdfs dfs -chmod -R 777 /user/fintech
hdfs dfs -mkdir -p /user/hive/warehouse
hdfs dfs -chmod 777 /user/hive/warehouse

echo "Uploading transaction data..."
hdfs dfs -put -f /tmp/spending_patterns_detailed.csv /user/fintech/transactions/

echo "Done. Verifying..."
hdfs dfs -ls /user/fintech/transactions/
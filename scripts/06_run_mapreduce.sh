#!/bin/bash
# Run Hadoop Streaming MapReduce job for per-category spend aggregation.
# Execute from project root: bash scripts/06_run_mapreduce.sh
set -e

echo "Copying mapper and reducer to namenode..."
docker cp scripts/06_mapreduce_mapper.py namenode:/tmp/
docker cp scripts/06_mapreduce_reducer.py namenode:/tmp/

echo "Submitting Hadoop Streaming job..."
docker exec namenode bash -c "
  chmod +x /tmp/06_mapreduce_mapper.py /tmp/06_mapreduce_reducer.py

  # remove previous output if exists
  hdfs dfs -rm -r -f /user/fintech/mr_category_agg/

  STREAMING_JAR=\$(find /opt/hadoop -name 'hadoop-streaming-*.jar' 2>/dev/null | head -1)

  hadoop jar \"\$STREAMING_JAR\" \
    -files /tmp/06_mapreduce_mapper.py,/tmp/06_mapreduce_reducer.py \
    -mapper  'python3 06_mapreduce_mapper.py' \
    -reducer 'python3 06_mapreduce_reducer.py' \
    -input   /user/fintech/transactions/spending_patterns_detailed.csv \
    -output  /user/fintech/mr_category_agg/
"

echo "MapReduce complete. Verifying output..."
docker exec namenode hdfs dfs -cat /user/fintech/mr_category_agg/part-00000 | head -20

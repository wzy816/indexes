# big data

## Conference

- SIGMOD/PODS
- SIGKDD
- SIGIR
- WSDM
- VLDB
- ICDE

http://www.conferenceranks.com/

## ETL

- [etl process overview](https://www.keboola.com/blog/etl-process-overview)
- [Building your own ETL platform](https://gtoonstra.github.io/etl-with-airflow/platform.html)

## Clickhouse

- [DMP 平台在贝壳的实践和应用](https://mp.weixin.qq.com/s?__biz=MzIyMTg0OTExOQ==&mid=2247485738&idx=1&sn=71d61dae19c6d6111e25e420207600f9&chksm=e8373a5adf40b34c072a79d6e05d2e1d87c7b78adfcabd30f8b0df09b3ccab79db13d2c4733c&scene=21#wechat_redirect)
- [Bitmap 用户分群在贝壳 DMP 的实践和应用](https://cloud.tencent.com/developer/article/1684659)
- [Clickhouse 源码导读](http://sineyuan.github.io/post/clickhouse-source-guide/)
- [ClickHouse 在字节广告 DMP& CDP 的应用](https://mp.weixin.qq.com/s/lYjIfKS8k9ZHPrxBRYOBrw)

## Data Mining

- [《李航 - 统计学习方法》学习笔记](https://windmissing.github.io/LiHang-TongJiXueXiFangFa/)
- [数据挖掘概览](https://wizardforcel.gitbooks.io/data-mining-book/content/)

## 三驾马车 & google

- big table :book:
- gfs :book:
- MapReduce :book:
- WEB SEARCH FOR A PLANET THE GOOGLE CLUSTER ARCHITECTURE :book:
- Borg, Omega, and Kubernetes :book:
- GOOGLE-WIDE PROFILING A CONTINUOUS PROFILING INFRASTRUCTURE FOR DATA CENTERS :book:
- MillWheel Fault-Tolerant Stream Processing at Internet Scale :book:
- Mesa Geo-Replicated, Near Real-Time, Scalable Data Warehousing :book:
- Goods: Organizing Google’s Datasets :book:
  extract metadata of billions of dataset
- Spanner: Google’s Globally Distributed Database
- Dapper, a Large-Scale Distributed Systems Tracing Infrastructure

## Distribute Systems

- distributed systems :book:
- A Note on Distributed Computing :book:
- fallacies of distributed computing explained :book:

## Elasticsearch

- [Mastering Elasticsearch（中文版）](https://doc.yonyoucloud.com/doc/mastering-elasticsearch/index.html)

## Flink & Streaming

- Flink 基础教程 :book:
- [Large Scale Stream Processing with Blink SQL at Alibaba](https://www.youtube.com/watch?v=-Q9VG5QwLzY) :movie_camera:
- Lightweight Asynchronous Snapshots for Distributed Dataflows :book:
- [Streaming 101: The world beyond batch](https://www.oreilly.com/radar/the-world-beyond-batch-streaming-101/)
- [Streaming 102: The world beyond batch](https://www.oreilly.com/radar/the-world-beyond-batch-streaming-102/)
- Realtime Data Processing at Facebook :book:
- State Management in Apache Flink :book:
- The Dataflow Model :book:

## HDFS

- Apache Hadoop 3.1.1 – HDFS Architecture :book:

## Hive

- [Language Manual](https://cwiki.apache.org/confluence/display/Hive/LanguageManual)

## Spark && Batch

- Learning Spark :book:
- Advanced Analytics with Spark :book:
- [Airstream: Spark Streaming At Airbnb](https://www.youtube.com/watch?v=tJ1uIHQtoNc) :book:
- mastering spark sql :book:
- [Introducing Window Functions in Spark SQL](https://databricks.com/blog/2015/07/15/introducing-window-functions-in-spark-sql.html)
- [Spark Window Function - PySpark](https://knockdata.github.io/spark-window-function/)

## Parquet

- [Parquet Logical Type Definitions](https://github.com/apache/parquet-format/blob/master/LogicalTypes.md)
- ColumnStores vs. RowStores How Different Are They Really :book:

## t-digest

- Computing Extremely Accurate Quantiles Using t-digests :book:
- [论文作者的 java 实现](https://github.com/tdunning/t-digest/blob/master/core/src/main/java/com/tdunning/math/stats/TDigest.java)
- [scala 实现](https://gist.github.com/RobColeman/c4c948f6365dc788a09d)
- [TDigestUDAF](https://github.com/isarn/isarn-sketches-spark/blob/master/src/main/scala/org/isarnproject/sketches/udaf/TDigestUDAF.scala)
- [有问题的 python 实现](https://github.com/CamDavidsonPilon/tdigest/blob/master/tdigest/tdigest.py)

## Practice

- 大数据之路：阿里巴巴大数据实践 :book:
- [有赞大数据平台安全建设实践](https://tech.youzan.com/bigdatasafety/)

## Snowflake

- The Snowflake Elastic Data Warehourse :book:
- Building An Elastic Query Engine on Disaggregated Storage :book:

## SQL

- Volcano-An Extensible and Parallel Query Evaluation System :book:

## FAISS

- A Survey of Product Quantization :book:

## OLAP & OLTP

- Kudu Storage for Fast Analytics on Fast Data :book:
- Impala A Modern, Open-Source SQL Engine for Hadoop :book:
- Dremel Interactive Analysis of Web-Scale Datasets :book:
- TiDB: A Raft-based HTAP Database, VLDB 2020
  sql engine for compute on top of storage = kv for oltp + parquet for olap, data is replicated
- Delta Lake: High-Performance ACID Table Storage over Cloud Object Stores, VLDB 2020
  table on OSS，use log、log checkpoint to achieve table's ACID
- Druid: A Real-time Analytical Data Store
- Vectorwise: Beyond Column Stores
- HyPer: A Hybrid OLTP&OLAP Main Memory Database System Based on Virtual Memory Snapshots
- Hekaton: SQL Server’s Memory-Optimized OLTP Engine

## F1

- F1 A Distributed SQL Database That Scales :book:

## Calcite

- Apache Calcite A Foundational Framework for Optimized Query Processing Over Heterogeneous Data Sources :book:

## chubby

- The Chubby lock service for loosely-coupled distributed systems :book:

## CockroachDB

- CockroachDB: The Resilient Geo-Distributed SQL Database :book:

## Hologres

- Alibaba Hologres: A Cloud-Native Service for Hybrid Serving/Analytical Processing :book:

## Kafka

- original design doc [Exactly Once Delivery and Transactional Messaging in Kafka](https://docs.google.com/document/d/11Jqy_GjUGtdXJK94XGsEIK7CP1SnQGdp2eF0wSw9ra8)
- [exactly once semantics](https://www.confluent.io/blog/exactly-once-semantics-are-possible-heres-how-apache-kafka-does-it/)
- [transactions apache kafka](https://www.confluent.io/blog/transactions-apache-kafka/)
- [enabling exactly once](https://www.confluent.io/blog/enabling-exactly-once-kafka-streams/)
- [EoS Abort Index Proposal](https://docs.google.com/document/d/1Rlqizmk7QCDe8qAnVW5e5X8rGvn6m2DCR3JR2yqwVjc/)

## Cassandra

- Cassandra - A Decentralized Structured Storage System

# big data

## Overview

- 大数据之路：阿里巴巴大数据实践 :book:
- [有赞大数据平台安全建设实践](https://tech.youzan.com/bigdatasafety/)
- [《李航 - 统计学习方法》学习笔记](https://windmissing.github.io/LiHang-TongJiXueXiFangFa/)
- [数据挖掘概览](https://wizardforcel.gitbooks.io/data-mining-book/content/)
- [etl process overview](https://www.keboola.com/blog/etl-process-overview)
- [Building your own ETL platform](https://gtoonstra.github.io/etl-with-airflow/platform.html)

### 三驾马车

- big table :book:
- gfs :book:
- MapReduce :book:

## Compute

### SQL

- Volcano-An Extensible and Parallel Query Evaluation System :book:
- Apache Calcite A Foundational Framework for Optimized Query Processing Over Heterogeneous Data Sources :book:

### Hadoop

- Apache Hadoop 3.1.1 – HDFS Architecture :book:
- [Hive Language Manual](https://cwiki.apache.org/confluence/display/Hive/LanguageManual)
- Learning Spark :book:
- Advanced Analytics with Spark :book:
- [Airstream: Spark Streaming At Airbnb](https://www.youtube.com/watch?v=tJ1uIHQtoNc) :book:
- mastering spark sql :book:
- [Introducing Window Functions in Spark SQL](https://databricks.com/blog/2015/07/15/introducing-window-functions-in-spark-sql.html)
- [Spark Window Function - PySpark](https://knockdata.github.io/spark-window-function/)

### Streaming

- Flink 基础教程 :book:
- [Large Scale Stream Processing with Blink SQL at Alibaba](https://www.youtube.com/watch?v=-Q9VG5QwLzY) :movie_camera:
- Lightweight Asynchronous Snapshots for Distributed Dataflows :book:
- [Streaming 101: The world beyond batch](https://www.oreilly.com/radar/the-world-beyond-batch-streaming-101/)
- [Streaming 102: The world beyond batch](https://www.oreilly.com/radar/the-world-beyond-batch-streaming-102/)
- Realtime Data Processing at Facebook :book:
- State Management in Apache Flink :book:
- The Dataflow Model :book:
- MillWheel Fault-Tolerant Stream Processing at Internet Scale :book:

### Snowflake

- The Snowflake Elastic Data Warehourse :book:
- Building An Elastic Query Engine on Disaggregated Storage :book:

## Storage

### Format

- [Parquet Logical Type Definitions](https://github.com/apache/parquet-format/blob/master/LogicalTypes.md)
- ColumnStores vs. RowStores How Different Are They Really :book:

### NewSQL / OLTP

- F1 A Distributed SQL Database That Scales :book:
- CockroachDB: The Resilient Geo-Distributed SQL Database :book:
- Alibaba Hologres: A Cloud-Native Service for Hybrid Serving/Analytical Processing :book:
- Kudu Storage for Fast Analytics on Fast Data :book:
- Impala A Modern, Open-Source SQL Engine for Hadoop :book:
- Dremel Interactive Analysis of Web-Scale Datasets :book:
- Delta Lake: High-Performance ACID Table Storage over Cloud Object Stores
  table on OSS，use log、log checkpoint to achieve table's ACID
- Druid: A Real-time Analytical Data Store
- Vectorwise: Beyond Column Stores
- HyPer: A Hybrid OLTP&OLAP Main Memory Database System Based on Virtual Memory Snapshots
- Hekaton: SQL Server’s Memory-Optimized OLTP Engine
- Cassandra - A Decentralized Structured Storage System

### OLAP

- [Clickhouse 源码导读](http://sineyuan.github.io/post/clickhouse-source-guide/)
- [ClickHouse 在字节广告 DMP & CDP 的应用](https://mp.weixin.qq.com/s/lYjIfKS8k9ZHPrxBRYOBrw)

### HTAP

- [TiDB: A Raft-based HTAP Database](https://www.vldb.org/pvldb/vol13/p3072-huang.pdf)
  - 强调 freshness（分析查询实时性） 和 isolation（事务查询和分析查询资源隔离）
  - 三个 component
    - distributed storage layer， 包括用于 OLTP 的行式存储（TiKV）和用于 OLAP 的列式存储（TiFlash）
    - compute layer，包括 TiSpark 和 SQL 引擎
    - placement driver, 用于管理 region、提供 timestamp oracle
  - 行式（leader）内部形成 raft group，行式到列式（learner）之间用 raft 算法异步获得数据
  - 行式存储 TiKV 用 LSM Tree，inspired by Google's Percolator， internally 用 RocksDB
  - 列式存储 TiFlash 基于 DeltaTree，顶层是 B+Tree，更新数据先 append 到 delta space，然后再 perioidcally merge 成大文件到 stable space

### Elasticsearch

- [Mastering Elasticsearch（中文版）](https://doc.yonyoucloud.com/doc/mastering-elasticsearch/index.html)

### Kafka

- original design doc [Exactly Once Delivery and Transactional Messaging in Kafka](https://docs.google.com/document/d/11Jqy_GjUGtdXJK94XGsEIK7CP1SnQGdp2eF0wSw9ra8)
- [exactly once semantics](https://www.confluent.io/blog/exactly-once-semantics-are-possible-heres-how-apache-kafka-does-it/)
- [transactions apache kafka](https://www.confluent.io/blog/transactions-apache-kafka/)
- [enabling exactly once](https://www.confluent.io/blog/enabling-exactly-once-kafka-streams/)
- [EoS Abort Index Proposal](https://docs.google.com/document/d/1Rlqizmk7QCDe8qAnVW5e5X8rGvn6m2DCR3JR2yqwVjc/)
- Kafka: a Distributed Messaging System for Log Processing :book:

## Distribute Systems

- distributed systems :book:
- A Note on Distributed Computing :book:
- fallacies of distributed computing explained :book:
- [The Chubby lock service for loosely-coupled distributed systems](https://research.google/pubs/pub27897/)
  - [The Chubby Lock Service notes](https://github.com/jguamie/system-design/blob/master/notes/chubby-lock-service.md)
- [Zookeeper: Wait-free coordination for Internet-scale systems](https://www.usenix.org/legacy/event/atc10/tech/full_papers/Hunt.pdf)
- Dapper, a Large-Scale Distributed Systems Tracing Infrastructure

## Dataset

- Goods: Organizing Google’s Datasets :book:
  - extract metadata of billions of dataset
- WEB SEARCH FOR A PLANET THE GOOGLE CLUSTER ARCHITECTURE :book:

## 三驾马车 & google

- Mesa Geo-Replicated, Near Real-Time, Scalable Data Warehousing :book:

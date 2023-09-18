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
- Alibaba Hologres: A Cloud-Native Service for Hybrid Serving/Analytical Processing :book:

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

## Storage

### Format

- [Parquet Logical Type Definitions](https://github.com/apache/parquet-format/blob/master/LogicalTypes.md)
- ColumnStores vs. RowStores How Different Are They Really :book:
- [Weaving Relations for Cache Performance](https://www.vldb.org/conf/2001/P169.pdf)
  - introduce new data layout (PAX) for RDS using fixed/variable-length minipage within page
  - inter-record spatial locality and high cache performance compared with NSM
  - less reconstruction cost compared with DSM
- [A Case for Fractured Mirrors](https://www.vldb.org/conf/2002/S12P03.pdf)

### KV

- leveldb
  - [API](https://github.com/google/leveldb/blob/main/doc/index.md)
  - single thread, log structured merge (lsm) architecture
  - use level-style compaction
- RocksDB
  - embeddable
  - use universal style compaction, reduce write amplifaction to < 10
  - bloom prefix scan reduce read amplication
- [Dynamo: Amazon’s Highly Available Key-value Store](https://assets.amazon.science/ac/1d/eb50c4064c538c8ac440ce6a1d91/dynamo-amazons-highly-available-key-value-store.pdf)
  - eventually consistant data store, always writable, conflicts are resolved at read
  - high availability, sacrifice consistency under certain failure scenarios
  - consistent hashing and virtual node for incremental scaling
  - use merkle tree for replica syncronization check
  - gossip-based protocal to maintain an eventually consistent view of membership

### NewSQL / OLTP / OLAP / HTAP

- CockroachDB: The Resilient Geo-Distributed SQL Database :book:
- [Kudu Storage for Fast Analytics on Fast Data](https://kudu.apache.org/kudu.pdf)
  - storage system btw hdfs and hbase, providing tables with schema and primary key
  - one centralized, replicated master, and many tablet servers
  - tables use horizontal partitioning, schema can be 0+ hash-partitioning and optional range-partitioning
  - use raft algorithm to replicate operation logs
  - use raft to backup master process, master is only an observer of dynamic cluster state
  - a hybrid columnar store for tablet storage
    - MemRowSets(MassTree), in-memory concurrent B+ tree with optimistic locking, for effcient scan over pk range or lookup
    - DiskRowSet, has base store and delta store (for update and delete records)
  - maintenance threads run all the time in tablet server
  - benchmark against parquest on TPC-H, against hbase on YCSB
- Impala A Modern, Open-Source SQL Engine for Hadoop :book:
- Dremel Interactive Analysis of Web-Scale Datasets :book:
- Delta Lake: High-Performance ACID Table Storage over Cloud Object Stores
  - table on OSS，use log、log checkpoint to achieve table's ACID
- Druid: A Real-time Analytical Data Store
- Vectorwise: Beyond Column Stores
- HyPer: A Hybrid OLTP&OLAP Main Memory Database System Based on Virtual Memory Snapshots
- Hekaton: SQL Server’s Memory-Optimized OLTP Engine
- Cassandra - A Decentralized Structured Storage System
- HBase
- Mesa Geo-Replicated, Near Real-Time, Scalable Data Warehousing :book:
- [Clickhouse 源码导读](http://sineyuan.github.io/post/clickhouse-source-guide/)
- [ClickHouse 在字节广告 DMP & CDP 的应用](https://mp.weixin.qq.com/s/lYjIfKS8k9ZHPrxBRYOBrw)
- [TiDB: A Raft-based HTAP Database](https://www.vldb.org/pvldb/vol13/p3072-huang.pdf)
  - 强调 freshness（分析查询实时性） 和 isolation（事务查询和分析查询资源隔离）
  - 三个 component
    - distributed storage layer， 包括用于 OLTP 的行式存储（TiKV）和用于 OLAP 的列式存储（TiFlash）
    - compute layer，包括 TiSpark 和 SQL 引擎
    - placement driver, 用于管理 region、提供 timestamp oracle
  - 行式（leader）内部形成 raft group，行式到列式（learner）之间用 raft 算法异步获得数据
  - 行式存储 TiKV 用 LSM Tree，inspired by Google's Percolator， internally 用 RocksDB
  - 列式存储 TiFlash 基于 DeltaTree，顶层是 B+Tree，更新数据先 append 到 delta space，然后再 perioidcally merge 成大文件到 stable space
- [F1 A Distributed SQL Database That Scales](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41344.pdf)
  - hybrid relational and NoSQL database, use spanner as low-level storage
  - protocol buffer used in structured type
  - three types of transactions, by default use mainly optimistic transactions (a read phase + a short write phase)

### Elasticsearch

- [Mastering Elasticsearch（中文版）](https://doc.yonyoucloud.com/doc/mastering-elasticsearch/index.html)

### Kafka

- original design doc [Exactly Once Delivery and Transactional Messaging in Kafka](https://docs.google.com/document/d/11Jqy_GjUGtdXJK94XGsEIK7CP1SnQGdp2eF0wSw9ra8)
- [exactly once semantics](https://www.confluent.io/blog/exactly-once-semantics-are-possible-heres-how-apache-kafka-does-it/)
- [transactions apache kafka](https://www.confluent.io/blog/transactions-apache-kafka/)
- [enabling exactly once](https://www.confluent.io/blog/enabling-exactly-once-kafka-streams/)
- [EoS Abort Index Proposal](https://docs.google.com/document/d/1Rlqizmk7QCDe8qAnVW5e5X8rGvn6m2DCR3JR2yqwVjc/)
- Kafka: a Distributed Messaging System for Log Processing :book:

### Snowflake

- [The Snowflake Elastic Data Warehourse](https://info.snowflake.net/rs/252-RFO-227/images/Snowflake_SIGMOD.pdf)
  - decouple storage and compute by two scalable services
  - 3-layer architecture
    1. data storage, on object storage
       - usability, scalability, HA
       - range GET
    2. virtual warehouse, on scalable MPP worker nodes
    3. cloud service, on ec2
       - concurrency control
       - min-max pruning
- [Building An Elastic Query Engine on Disaggregated Storage](https://www.usenix.org/system/files/nsdi20-paper-vuppalapati.pdf)
  - local ephemeral storage as write-through cache

## Distribute Systems

- distributed systems :book:
- A Note on Distributed Computing :book:
- fallacies of distributed computing explained :book:
- [The Chubby lock service for loosely-coupled distributed systems](https://research.google/pubs/pub27897/)
  - [The Chubby Lock Service notes](https://github.com/jguamie/system-design/blob/master/notes/chubby-lock-service.md)
- [Zookeeper: Wait-free coordination for Internet-scale systems](https://www.usenix.org/legacy/event/atc10/tech/full_papers/Hunt.pdf)
- Dapper, a Large-Scale Distributed Systems Tracing Infrastructure
- Technical Report: HybridTime - Accessible Global Consistency with High Clock Uncertainty

## Dataset

- Goods: Organizing Google’s Datasets :book:
  - extract metadata of billions of dataset
- WEB SEARCH FOR A PLANET THE GOOGLE CLUSTER ARCHITECTURE :book:

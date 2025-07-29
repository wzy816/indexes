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
- [Impala: A Modern, Open-Source SQL Engine for Hadoop](https://www.cidrdb.org/cidr2015/Papers/CIDR15_Paper28.pdf)
- [Hekaton: SQL Server’s Memory-Optimized OLTP Engine](https://www.microsoft.com/en-us/research/wp-content/uploads/2013/06/Hekaton-Sigmod2013-final.pdf)

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
  - one copy of DSM and one copy of DSM, declustered and stored on the same disk, being logically identical

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

- [CockroachDB: The Resilient Geo-Distributed SQL Database](https://dl.acm.org/doi/pdf/10.1145/3318464.3386134)
  - SQL DBMS for OLTP, resilient, shared-nothing architecture
  - layered architecture inside node
    - SQL engine
    - transactional KV
    - key space data chunk, aka 64MB Ranges, has range-partitioning maintained by a two-level index structure
    - replication, 3 replica, use raft consesus on low-level edit commands
    - stored on disk-backed RockDB
  - range-partitioning, two-level indexing structure
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
- [Delta Lake: High-Performance ACID Table Storage over Cloud Object Stores](https://www.vldb.org/pvldb/vol13/p3411-armbrust.pdf)
  - ACID table on OSS
  - use a write-ahead log compacted into parquet stored in oss to provide ACID property
  - data objects and log records with checkpoints
- [Druid: A Real-time Analytical Data Store](http://static.druid.io/docs/druid.pdf)
- [Vectorwise: Beyond Column Stores](http://sites.computer.org/debull/A12mar/vectorwise.pdf)
- [HyPer: A Hybrid OLTP&OLAP Main Memory Database System Based on Virtual Memory Snapshots](https://cs.brown.edu/courses/cs227/archives/2012/papers/olap/hyper.pdf)
- [Cassandra - A Decentralized Structured Storage System](https://www.cs.cornell.edu/projects/ladis2009/papers/lakshman-ladis2009.pdf)
- HBase
- [Mesa Geo-Replicated, Near Real-Time, Scalable Data Warehousing](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42851.pdf)
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

### ClickHouse

- [ByteHouse](https://www.cnblogs.com/bytedata/p/17797465.html)
- 源码解读
  - [write MergeTree](https://zhuanlan.zhihu.com/p/460000230)
  - [server](https://sineyuan.github.io/post/clickhouse-source-guide/)
  - [ClickHouse 和他的朋友们](https://bohutang.me/2020/06/05/clickhouse-and-friends-development/)
  - [ClickHouse 和他的朋友们（5）存储引擎技术进化与 MergeTree](https://bohutang.me/2020/06/20/clickhouse-and-friends-merge-tree-algo/)
  - [ClickHouse 和他的朋友们（6）MergeTree 存储结构](https://bohutang.me/2020/06/26/clickhouse-and-friends-merge-tree-disk-layout/)
- vector search
  - [p1](https://clickhouse.com/blog/vector-search-clickhouse-p1)
  - [p2](https://clickhouse.com/blog/vector-search-clickhouse-p2)
- [Clickhouse 源码导读](http://sineyuan.github.io/post/clickhouse-source-guide/)
- [ClickHouse 在字节广告 DMP & CDP 的应用](https://mp.weixin.qq.com/s/lYjIfKS8k9ZHPrxBRYOBrw)

### Milvus

- [paper](https://www.cs.purdue.edu/homes/csjgwang/pubs/SIGMOD21_Milvus.pdf)
- deep dive
  - <https://milvus.io/blog/deep-dive-1-milvus-architecture-overview.md>
  - <https://milvus.io/blog/deep-dive-2-milvus-sdk-and-api.md>
  - <https://milvus.io/blog/deep-dive-3-data-processing.md>
  - <https://milvus.io/blog/deep-dive-4-data-insertion-and-data-persistence.md>
  - <https://milvus.io/blog/deep-dive-5-real-time-query.md>
  - <https://milvus.io/blog/deep-dive-6-oss-qa.md>
  - <https://milvus.io/blog/deep-dive-7-query-expression.md>
  - <https://milvus.io/blog/deep-dive-8-knowhere.md>
- DiskANN
  - [paper](https://suhasjs.github.io/files/diskann_neurips19.pdf)
  - [Paper Reading | DiskANN： 十亿规模数据集上高召回高 QPS 的 ANNS 单机方案](https://cloud.tencent.com/developer/article/1865556)
  - [kimi](kimi/DiskANN:%20Fast%20Accurate%20Billion-point%20Nearest%20Neighbor%20Search%20on%20a%20Single%20Node.md)
  - [github](https://github.com/microsoft/DiskANN)

## Distribute Systems

- distributed systems :book:
- [A Note on Distributed Computing](https://scholar.harvard.edu/files/waldo/files/waldo-94.pdf)
  - [kimi](kimi/A%20Note%20on%20Distributed%20Computing.md)
- fallacies of distributed computing explained :book:
- [The Chubby lock service for loosely-coupled distributed systems](https://research.google/pubs/pub27897/)
  - [The Chubby Lock Service notes](https://github.com/jguamie/system-design/blob/master/notes/chubby-lock-service.md)
- [Zookeeper: Wait-free coordination for Internet-scale systems](https://www.usenix.org/legacy/event/atc10/tech/full_papers/Hunt.pdf)
- Dapper, a Large-Scale Distributed Systems Tracing Infrastructure
- Technical Report: HybridTime - Accessible Global Consistency with High Clock Uncertainty
- [The Distributed Reader](https://reiddraper.github.io/distreader/)
- [A Distributed Systems Reading List](https://dancres.github.io/Pages/)
- [学习分布式系统需要怎样的知识？](https://www.zhihu.com/question/23645117/answer/108171462)

### Causality

- [Time, Clocks, and the Ordering of Events in a Distributed System](https://lamport.azurewebsites.net/pubs/time-clocks.pdf)
  - [kimi](kimi/Time,%20Clocks,%20and%20the%20Ordering%20of%20Events%20in%20a%20Distributed%20System.md)
  - [解读](http://zhangtielei.com/posts/blog-time-clock-ordering.html)
- [Interval Tree Clocks: A Logical Clock for Dynamic Systems](https://gsd.di.uminho.pt/members/cbm/ps/itc2008.pdf)
  - [kimi](kimi/Interval%20Tree%20Clocks:%20A%20Logical%20Clock%20for%20Dynamic%20Systems.md)

### Consistency

- [条分缕析分布式：到底什么是一致性？](http://zhangtielei.com/posts/blog-distributed-consistency.html)
- [条分缕析分布式：浅析强弱一致性](http://zhangtielei.com/posts/blog-distributed-strong-weak-consistency.html)

### Consensus

- [Towards Robust Distributed Systems](https://sites.cs.ucsb.edu/~rich/class/cs293-cloud/papers/Brewer_podc_keynote_2000.pdf)
  - Consistency, Availability, Partitioning
  - for PostgreSQL with ACID, prefer c over a
  - for Basic Availability, Soft-state, Eventual consistency,like mongodb, prefer a over c
- [Consistency Tradeoffs in Modern Distributed Database System Design](https://www.cs.umd.edu/~abadi/papers/abadi-pacelc.pdf)
  - for system with partition, tradeoff btw consistency and availability
  - for system with no partition, tradeoff btw consistency and latency
- [In Search of an Understandable Consensus Algorithm](https://raft.github.io/raft.pdf)
  - [kimi](kimi/In%20Search%20of%20an%20Understandable%20Consensus%20Algorithm.md)

## Dataset

- Goods: Organizing Google’s Datasets :book:
  - extract metadata of billions of dataset
- WEB SEARCH FOR A PLANET THE GOOGLE CLUSTER ARCHITECTURE :book:
- [Dremel Interactive Analysis of Web-Scale Datasets](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/36632.pdf)
  - [kimi](kimi/Dremel%20Interactive%20Analysis%20of%20Web-Scale%20Datasets.md)

## Volcano

- [helm charts](https://github.com/volcano-sh/helm-charts/releases)
- [解锁 Kubernetes 批处理新范式：Volcano 调度引擎初体验](https://www.lixueduan.com/posts/kubernetes/43-volcano-simple-usage/)

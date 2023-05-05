# snowflake

A multi-cluster, shared-data data warehouse on cloud solution.

## 3-layer architecture

from bottom to top

1. data storage on object storage
2. virtual warehouse on scalable MPP worker nodes
3. data service on ec2: access control, query optimization, etc

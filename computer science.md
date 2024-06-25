# computer science

## Principles

- 编码：隐匿在计算机软硬件背后的语言 :book:

## Compiler

- [antlr mega tutorial](https://tomassetti.me/antlr-mega-tutorial/#lexers-and-parser)
- [Notes on Formal Language Theory and Parsing](https://ricknouwen.org/d/lecture-notes-formal-languages-nouwen.pdf)
  - Chomsky's grammar hierachy
  - parsing algorithm
    1. use a FILO stack to store rule elements
    2. pop element, if nonterminal, push new element(aka derivation), if match, advance
    3. if stack is empty, accept
  - generate a parser:
    - each production rule as defining a function named by the nonterminal
    - see nonterminal, call the function
    - see terminal, match it with input symbol
- [LL and LR Parsing Demystified](https://blog.reverberate.org/2013/07/ll-and-lr-parsing-demystified.html)
- [nearley](https://github.com/kach/nearley)
- Compilers Principles, Techniques, and Tools :book:
- EBNF: A Notation to Describe Syntax :book:
- [Basics of Compiler Design](http://hjemmesider.diku.dk/~torbenm/Basics/basics_lulu2.pdf) :book:
- Parsing Techniques: A Practical Guide

## interlang / artlang / conlang

- Toki Pona
  - [conlang critic](https://www.youtube.com/watch?v=eLn6LC1RpAo&ab_channel=janMisali)
  - easy to learn
- lojban
  - [The Complete Lojban Language](https://lojban.org/publications/cll/cll_v1.1_book.pdf)
    - [The Lojban Reference Grammar](https://lojban.github.io/cll/)
  - [conlang critic](https://www.youtube.com/watch?v=l-unefmAo9k&ab_channel=janMisali)
- [toaq](https://toaq.me/Main_Page)
- loglish
  - [Loglish](https://www.goertzel.org/new_research/Loglish.htm)

## Versioning

- [Semantic Versioning 2.0.0](https://semver.org/)

## Ontology

- Ontology Development 101: A Guide to Creating Your First Ontology :book:
- Ontology Representation & Querying for Realizing Semantics-driven Applications 📖
- Who's afraid of Ontology? :book:
- [ER Diagram](https://www.visual-paradigm.com/cn/guide/data-modeling/what-is-entity-relationship-diagram/)

## Linux

- linux 内核设计与实现
- 深入理解 linux 内核
- linux 内核源代码景分析

## Network

- 《Computer Networks, Fifth Edition - A Systems Approach》
- [network fundamentals](https://www.youtube.com/playlist?list=PLDQaRcbiSnqF5U8ffMgZzS7fq1rHUI3Q8) 🎥

## Database

- [VLL: a lock manager redesign for main memory database systems](https://www.cs.umd.edu/~abadi/papers/vldbj-vll.pdf)
- [Foundations of Databases](https://wiki.epfl.ch/provenance2011/documents/foundations+of+databases-abiteboul-1995.pdf)
- [存储引擎数据结构优化 (1):cpu bound](https://www.douban.com/note/304123656/)
- [存储引擎数据结构优化 (2):io bound](https://www.douban.com/note/304349195/)
- [“写优化”的数据结构 (1)：AOF 和 b-tree 之间](https://www.douban.com/note/269741273/)
- [“写优化”的数据结构 (2)：buffered tree](https://www.douban.com/note/269744617/)
- [“写优化”的数据结构 (3)：small-splittable-tree](https://www.douban.com/note/269750379/)
- [Data Structures and Algorithms for Big Databases](https://www3.cs.stonybrook.edu/~bender/talks/2013-BenderKuszmaul-xldb-tutorial.pdf)
- [B-Tree vs LSM-Tree](https://tikv.org/deep-dive/key-value-engine/b-tree-vs-lsm/)
- [Algorithms for Recovery and Isolation Exploiting Semantics](https://en.wikipedia.org/wiki/Algorithms_for_Recovery_and_Isolation_Exploiting_Semantics)

## ROS

- <http://wiki.ros.org/ROS>
- <https://docs.ros.org/en/galactic/Tutorials.html>
- <https://github.com/autovia/ros_hadoop>
  - java, 在 hadoop 中直接处理 bag 的一种方法，定义了  rosbaginputformat, 通过生成  idx.bin 索引作为 spark job 的配置文件
- <https://github.com/event-driven-robotics/importRosbag>
  - python, 对 bag reader 的过程式简化重写
- :star <https://github.com/cruise-automation/rosbag.js>
  - js 版 bag reader/writer, 可解析 message 数据，实现得较完整
- ? <https://github.com/facontidavide/ros_msg_parser>
  - c++,bag inspection tool, alternative for python rqt_plot, rqt_bag and rosbridge
- ? <https://github.com/tu-darmstadt-ros-pkg/cpp_introspection>
  - c++, inspection tool
- <https://github.com/Kautenja/rosbag-tools>
  - python3, 基于 rosbag 库，图片处理、视频处理
- <https://github.com/aktaylor08/RosbagPandas>
  - python2, 基于 rosbag 库提供了读取 rosbag 生成 pandas dataframe 的方法  bag_to_dataframe 生成 dataframe 数据
- <https://github.com/jmscslgroup/bagpy>
  - python, 基于 rosbag 库提供对 bag reader 的 wrapper, 对于预设的 msg 结构简化读取和可视化 msg
- <https://github.com/AtsushiSakai/rosbag_to_csv>
  - python, 基于 rosbag 库基于 read_messages 方法提取 topic，封装了 QtGui
- <https://github.com/IFL-CAMP/tf_bag>
  - python, 基于 rosbag 库提供 BagTfTransformer 类操作 tf 消息，可用 rosdep 安装
- <https://github.com/ToniRV/mesh_rviz_plugins>
  - rviz 插件显示 mesh
- <https://github.com/cruise-automation/webviz>
  - rviz in browser

## C / C++

- Expert C Programming - Deep C Secrets :book:
- [core guidelines](https://isocpp.github.io/CppCoreGuidelines/)
- [standards](https://isocpp.org/std/status)

## Golang

- [Go 语言高级编程](https://chai2010.cn/advanced-go-programming-book/)
- [Go with Versions](https://www.youtube.com/watch?v=F8nrpe0XWRg&list=PLq2Nv-Sh8EbbIjQgDzapOFeVfv5bGOoPE&index=3&t=0s) :movie_camera:
- [Effective Go - The Go Programming Language](https://golang.org/doc/effective_go.html)

## Scala

- [sbt](https://www.scala-sbt.org/1.x/docs/zh-cn/index.html)
- [写点什么](https://hongjiang.info/scala/)
- [opinionated scala](https://github.com/ghik/opinionated-scala) :book:

## Python

- [magic methods](https://github.com/RafeKettler/magicmethods)
- [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)
- [Full Grammar specification](https://docs.python.org/3/reference/grammar.html)
- [Time Complexity](https://wiki.python.org/moin/TimeComplexity)
- Problem Solving with Algorithms and Data Structures :book:
- Python Programming on Win32 :book:
- Cython :book:

## Javascript & Typescript & Node

- [es6 features](http://es6-features.org/#Constants)
- [event order](https://www.quirksmode.org/js/events_order.html#link4)
- [Data Structures in JavaScript](https://github.com/benoitvallon/computer-science-in-javascript/tree/master/data-structures-in-javascript)
- [You Don't Know JS Yet (book series) - 2nd Edition](https://github.com/getify/You-Dont-Know-JS) :book:
- [deep into node](https://yjhjstz.gitbooks.io/deep-into-node/content/)
- [What the f\*ck JavaScript?](https://github.com/denysdovhan/wtfjs)
- [Typescript Deep Dive](https://basarat.gitbook.io/typescript/)
- NODE.JS 入门 :book:
- <https://30secondsofcode.org/#minn>
- <https://github.com/stephentian/33-js-concepts>
- [Eloquent JavaScript](https://eloquentjavascript.net/)

## Latex

- cheatsheet :book：

## SQL

- [BNF Grammars for SQL-92, SQL-99 and SQL-2003](https://github.com/ronsavage/SQL)

## Git

- [Git Best Practices](https://gist.github.com/pandeiro/1552496)
- version control with git :book:

## Algorithm & OI

- An O(1) algorithm for implementing the LFU cache eviction scheme :book:
- [数据结构与算法之美](https://datastructure.xiaoxiaoming.xyz/#/README)
- 背包问题九讲 :book:
- [OI Wiki](https://oi-wiki.org/basic/)
- [algorithm part 1](https://www.coursera.org/learn/algorithms-part1)
- introduction to algorithms
- An Introduction to the Analysis of Algorithms
- Handbook of Data Structures and Applications
- 《编程珠玑》
- 《算法艺术与信息学竞赛算法》
- 《算法竞赛入门经典》
- Purely Functional Data Structures
- Pearls of Functional Algorithm Design
- Distributed Algorithm: An Intuitive Approach
- Hacker's Delight
- Algorithms on Strings
- Foundations of Multidimensional and Metric Data Structures(The Morgan Kaufmann Series in Computer Graphics)
- Clever Algorithms: Nature-inspired Programming Recipes
- [Hello 算法](https://www.hello-algo.com/chapter_preface/)
- [Algorithm by liuzhenglaichn](https://liuzhenglaichn.gitbook.io/algorithm/)

## t-digest

- Computing Extremely Accurate Quantiles Using t-digests :book:
- [论文作者的 java 实现](https://github.com/tdunning/t-digest/blob/master/core/src/main/java/com/tdunning/math/stats/TDigest.java)
- [scala 实现](https://gist.github.com/RobColeman/c4c948f6365dc788a09d)
- [TDigestUDAF](https://github.com/isarn/isarn-sketches-spark/blob/master/src/main/scala/org/isarnproject/sketches/udaf/TDigestUDAF.scala)
- [有问题的 python 实现](https://github.com/CamDavidsonPilon/tdigest/blob/master/tdigest/tdigest.py)

## math

### linear algebra

- 《LINEAR ALGEBRA FOR EVERYONE》Gilbert Strang 下载在 libgen
  - best
  - [the art of linear algebra](https://github.com/kenjihiranabe/The-Art-of-Linear-Algebra)
- [MIT 课程线性代数的笔记](https://github.com/zlotus/notes-linear-algebra)
- 《Linear Algebra and Its Application》Gilbert Strang

## UI Interaction

- [User Interface Analysis and Design](https://www.slideshare.net/SaqibRaza21/user-interface-analysis-and-design) :book:
- Reactive Vega: A Streaming Dataflow Architecture for Declarative Interactive Visualization :book:
- Voyager 2: Augmenting Visual Analysis with Partial View Specifications :book:
- Mental Models, Visual Reasoning and Interaction in Information Visualization: A Top-down Perspective :book:
- [Toward a Deeper Understanding of the Role of Interaction in Information Visualization](https://www.cc.gatech.edu/gvu/ii/talks/infovis07-interaction.pdf)
- A taxonomy of tools that support the fluent and flexible use of visualizations :book:
  - 12 visual analysis tasks
- Graphical Perception: Theory, Experimentation, and Application to the Development of Graphical Methods :book:
- Empirical Studies in Information Visualization: Seven Scenarios :book:
  - help make right evaluation goal, pick right question

## computer graphic

- [书单](https://github.com/GraphiCon/-)
- [BooksToRead](http://peterwonka.net/Documentation/BooksToRead.htm)
- [study path for game programmers](https://github.com/miloyip/game-programmer/)
- Fully Dynamic Constrained Delaunay Triangulations :book:
- Constrained Delaunay Triangulations :book:
- Theoretically Guaranteed Delaunay Mesh Generation :book:
- Learning OpenCV
  - 适合对计算机视觉和图像处理有基本了解的人

## game

- [Game Programming Patterns](https://gameprogrammingpatterns.com/)

## gltf

- gltf overview :book:
- [specification 1.0](https://github.com/KhronosGroup/glTF/tree/master/specification/1.0)
- [gltf tutorial](https://github.com/KhronosGroup/glTF-Tutorials/blob/master/gltfTutorial/README.md)

## canvas & svg

- Core HTML5 Canvas Graphics, Animation, and Game Development :book:
- animation svg :book:
- [anime.js](https://github.com/juliangarnier/anime)

## conference

- [ranks](http://www.conferenceranks.com/)
- SIGMOD/PODS
  - [2022 industrial](https://2022.sigmod.org/sigmod_industrial_list.shtml)
  - [2021 industrial](https://2021.sigmod.org/sigmod_industrial_list.shtml)
- SIGKDD
  - [2022](https://kdd.org/kdd2022/toc.html)
  - [2021](https://kdd.org/kdd2021/accepted-papers/index)
- SIGIR
- WSDM
- VLDB
  - [2022 industrial](https://vldb.org/2022/?papers-industrial)
  - [2021 industrial](https://vldb.org/2021/?papers-industrial)
- ICDE -[2022 industry](https://icde2022.ieeecomputer.my/accepted-industry-track/)
- ICR
- ITSC
- IROS
- NIPS: Neural Information Processing Systems <https://nips.cc/>
- ICML: International Conference on Machine Learning <https://icml.cc>
- UAI(AUAI): Association for Uncertainty in Artifical Intelligence <http://www.auai.org/>
- AISTATS: Artificial Intelligence and Statistics <http://www.aistats.org/>
- JMLR: Journal of Machine Learning Research <http://jmlr.org/>
- IJCAI: International Joint Conference on Artifical Intelligence <http://ijcai.org/>
- AAAI: Association for the Advancement of Aritifical Intelligence <http://www.aaai.org/home.html>

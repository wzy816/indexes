# system

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
- [存储引擎数据结构优化(1):cpu bound](https://www.douban.com/note/304123656/)
- [存储引擎数据结构优化(2):io bound](https://www.douban.com/note/304349195/)
- [“写优化”的数据结构(1)：AOF 和 b-tree 之间](https://www.douban.com/note/269741273/)
- [“写优化”的数据结构(2)：buffered tree](https://www.douban.com/note/269744617/)
- [“写优化”的数据结构(3)：small-splittable-tree](https://www.douban.com/note/269750379/)
- [Data Structures and Algorithms for Big Databases](https://www3.cs.stonybrook.edu/~bender/talks/2013-BenderKuszmaul-xldb-tutorial.pdf)
- [B-Tree vs LSM-Tree](https://tikv.org/deep-dive/key-value-engine/b-tree-vs-lsm/)
- [Algorithms for Recovery and Isolation Exploiting Semantics](https://en.wikipedia.org/wiki/Algorithms_for_Recovery_and_Isolation_Exploiting_Semantics)

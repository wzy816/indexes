# system

## Principles

- 编码：隐匿在计算机软硬件背后的语言 :book:

## Compiler

- [antlr mega tutorial](https://tomassetti.me/antlr-mega-tutorial/#lexers-and-parser)
- Notes on Formal Language Theory and Parsing :book:
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

## Versioning

- [Semantic Versioning 2.0.0](https://semver.org/)

## Ontology

- Ontology Development 101: A Guide to Creating Your First Ontology :book:
- Ontology Representation & Querying for Realizing Semantics-driven Applications :book:
- Who's afraid of Ontology? :book:

## Linux

- linux 内核设计与实现
- 深入理解 linux 内核
- linux 内核源代码景分析

## network

- 《Computer Networks, Fifth Edition - A Systems Approach》

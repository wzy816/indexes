# nlp

## Resources

- An Introduction to Information Retrieval :book:
  - must read chapter 2 and 6
- [Deep Learning for NLP resources](https://github.com/andrewt3000/dl4nlp)
- syntactic_structures :book:
- Natural Language Processing With Python :book:

## Word Embedding Models

### word2vec

An algorithm used to produce distributed representations of words.
It is NOT a deep neural network. The purpose is to group the vectors of similar words together in vector space. The output is a lookup dictionary where each item has a vector attached to it. Two methods of obtaining vector:

1. using context to predict a target word (a method known as continuous bag of words, or COW);
2. using a word to predict a target context, which is called skip-gram

#### Implementation

- [fasttext](https://github.com/facebookresearch/fastText#full-documentation)
- gensim

#### Tutorials

- word2vec Parameter Learning Explained
  - explained CBOW, CSG, hierachical softmax, subsampling
- [A Beginner's Guide to Word2Vec and Neural Word Embeddings](https://wiki.pathmind.com/word2vec)
- [Word2Vec Tutorial - The Skip-Gram Model](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
- [The Illustrated Word2vec](http://jalammar.github.io/illustrated-word2vec/)
- [tensorflow core word2vec](https://www.tensorflow.org/tutorials/text/word2vec)

### GloVe

Use word-to-word co-occurrence matrix. Evaluation methods includes word analogy tasks, word similarity tasks and named entity recognition. Better than word2vec.

- GloVe: Global Vectors for Word Representation :book:
- [website](https://nlp.stanford.edu/projects/glove/)

### Others

- Efficient Estimation of Word Representations in Vector Space :book:
- A Neural Probabilistic Language Model :book:
- Distributed Representations of Words and Phrases and their Compositionality
  - proposed hierachical softmax and negative sampling
- An empirical study of smoothing techniques for language modeling :book:
- Deep contextualized word representation :book:
  - propose ELMO
- BERT: Pre-training of Deep Bidirectional Transformers for Languange Understanding :book:

## Word Segmentation

### Papers

- 现代汉语语料库加工规范-词语切分及词性标注 :book:
- 汉语自动分词研究评述 :book:
- Which is More Suitable for Chinese Word Segmentation, the Generative Model or the Discriminative One? :book:
- 互联网时代的社会语言学：基于 SNS 的文本数据挖掘 :book:

### Algorithms

#### 基于匹配/机械分析

1. forward maximal matching MM 正向最大匹配法，最长词优先匹配法
2. reverse maximum matching RMM 逆向最大匹配法
3. bidirectional maximal matching 双向最大匹配
4. 最少分词法
5. 全切分法
6. [MMSEG](http://technology.chtsai.org/mmseg/)

1 和 2 速度快，歧义检测和消解步骤合一，但是需要完备的词典；3 结合 MM 和 RMM，实用性好，但是存在着切分歧义检测盲区；6 优化了最大匹配，基于一个词典，两个匹配规算法，四个消除歧义规则。

#### 基于统计

基本思想是相同上下文构成稳定的词的组合。一般与基于匹配的方法结合。

1. N-gram
2. Hidden Markov Model(HMM)
3. Maximum Entropy, ME
4. Conditional Random Fields，CRF

2 和 4 基于序列标注。

#### 基于语义（理解）的

1. Augmented Transition Network

#### 基于神经网络的

1. LSTM+CRF
2. BiLSTM+CRF

### Library

- [HanLP](https://github.com/hankcs/HanLP)
- [jieba](https://github.com/fxsjy/jieba)
- [swcs](http://www.xunsearch.com/scws/docs.php)
- [paoding-analysis](https://gitee.com/zhzhenqin/paoding-analysis)
- [pangusegment](https://archive.codeplex.com/?p=pangusegment)
- [SnowNLP](https://github.com/isnowfy/snownlp)
- [THULAC 清华](http://thulac.thunlp.org/)
- [FudanNLP 复旦](https://github.com/FudanNLP/fnlp)
- [fastNLP](https://github.com/fastnlp/fastNLP)
- [LTP 哈工大语言云](http://www.ltp-cloud.com/document2#api2_python_interface)
- [NLPIR 北理工](https://github.com/NLPIR-team/NLPIR)
- [StanfordCoreNLP](https://github.com/stanfordnlp/CoreNLP)
- [pkuseg 北大](https://github.com/lancopku/pkuseg-python)
- [baidu LAC](https://github.com/baidu/lac)
- [mmseg](https://pypi.org/project/mmseg/)

commercial

- [Boson NLP](http://docs.bosonnlp.com/tag.html)
- [Baidu NLP](https://cloud.baidu.com/doc/NLP/NLP-Python-SDK.html#.3C.CF.E9.5F.9C.E3.C3.45.DA.9C.9E.4C.F8.55.F1.E6)
- [Aliyun](https://help.aliyun.com/document_detail/61384.html?spm=a2c4g.11186623.6.549.27433020OdUf5E)
- [Sougou](http://www.sogou.com/labs/webservice/)
- [Tencent NLP](https://nlp.qq.com/help.cgi?topic=api#analysis)

### Evaluation Standard

[icwb2-data](http://sighan.cs.uchicago.edu/bakeoff2005/)

## Text Classification

- Convolutional Neural Networks for Sentence Classification :book:
- Character-level Convolutional Networks for Text Classification :book:
- Bag of Tricks for Efficient Text Classification :book:

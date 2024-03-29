# transformer

![](transformer/The-evolutionary-tree-of-modern-Large-Language-Models-LLMs-traces-the-development-of.png)

## General

### AIGC

- [A Comprehensive Survey of AI-Generated Content (AIGC): A History of Generative AI from GAN to ChatGPT](https://arxiv.org/pdf/2303.04226.pdf)
- [Google "We Have No Moat, And Neither Does OpenAI"](https://www.semianalysis.com/p/google-we-have-no-moat-and-neither)
- A Survey of Large Language Model
  - [paper en](https://arxiv.org/pdf/2303.18223.pdf)
  - [paper cn](https://github.com/RUCAIBox/LLMSurvey/blob/main/assets/LLM_Survey_Chinese_0418.pdf)
  - [github](https://github.com/RUCAIBox/LLMSurvey)
- [Planning for AGI and beyond by Sam Altman](https://openai.com/blog/planning-for-agi-and-beyond)
- [ChatGPT 原理介绍：从语言模型走近 chatgpt](https://zhuanlan.zhihu.com/p/608047052)
- [通向 AGI 之路：大型语言模型（LLM）技术精要](https://zhuanlan.zhihu.com/p/597586623)
  - LLM 作为交互接口理解人
  - 模型大小与训练数据量关系 => 涌现
  - 分步拆解，增强了推理 CoT 能力
  - 未来 LLM 模型稀疏化
- [State of GPT](https://build.microsoft.com/en-US/sessions/db3f4859-cd30-4445-a0cd-553c3304f8e2)
- [A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf)
  - Mode-1 (perception-action) and Mode-2 (model-predictive control, MPC)
  - propose a Joint Embedding Predictive Architectures for Self-Supervised Learning
- [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)
  - general methods that leverage computation are ultimately the most effective
- [AI engineer](https://www.latent.space/p/ai-engineer)
  - LLM enabled Product
  - software 3.0
- [Levels of AGI: Operationalizing Progress on the Path to AGI](https://arxiv.org/pdf/2311.02462.pdf)
  - define 6 criteria of AGI and 6-level ontology of AGI
- [THE IMPACT OF DEPTH AND WIDTH ON TRANSFORMER LANGUAGE MODEL GENERALIZATION](https://arxiv.org/pdf/2310.19956.pdf)

## Basic

### Transformer & Attention

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
- [Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
- [A Simple Example of Causal Attention Masking in Transformer Decoder](https://medium.com/@jinoo/a-simple-example-of-attention-masking-in-transformer-decoder-a6c66757bc7d)
- perceiver io
  - perform self-attention on latent variables, cross-attention on inputs, solve qudratic scaling of seq length
  - [paper](https://arxiv.org/pdf/2107.14795.pdf)
  - [hf doc](https://huggingface.co/docs/transformers/model_doc/perceiver)
  - [deepmind jax implementation](https://github.com/deepmind/deepmind-research/blob/master/perceiver/README.md)
  - [pytorch implementation](https://github.com/krasserm/perceiver-io)

### Long-Range attention

- <https://huggingface.co/blog/long-range-transformers>
  - 4 improvements on vanilla attention
- [Efficient Transformers: A Survey](https://arxiv.org/pdf/2009.06732.pdf)
  - chapter 2 very canonical
- [LONG RANGE ARENA: A BENCHMARK FOR EFFICIENT TRANSFORMERS](https://arxiv.org/pdf/2011.04006.pdf)
  - benchmark for evaluating model quality under long-context scenario
  - [github](https://github.com/google-research/long-range-arena)
- Linformer
  - low-rank, add projection to v & k
  - [paper](https://arxiv.org/pdf/2006.04768.pdf)
  - [Johnson–Lindenstrauss lemm proof](https://cseweb.ucsd.edu/~dasgupta/papers/jl.pdf)
  - [blog](https://tevenlescao.github.io/blog/fastpages/jupyter/2020/06/18/JL-Lemma-+-Linformer.html)
  - [hf blog](https://huggingface.co/blog/long-range-transformers#linformer-self-attention-with-linear-complexity)
- BigBird
  - block sparse attention up to 4096
  - [hf blog](https://huggingface.co/blog/big-bird#bigbird-block-sparse-attention)
  - [hf code](https://github.com/huggingface/transformers/blob/main/src/transformers/models/big_bird/modeling_big_bird.py#LL514C10-L514C10)

### Rotary Position Encoding

[notebook](transformer/rotary%20position%20embedding.ipynb)

- [blog CN , key equation 11 and 13](https://kexue.fm/archives/8265)
- [paper, aligned with blog CN](https://arxiv.org/pdf/2104.09864v4.pdf)
- [roformer github](https://github.com/ZhuiyiTechnology/roformer)
- [blog EN from eleuther AI](https://blog.eleuther.ai/rotary-embeddings/)

### Alibi

[notebook](transformer/alibi.ipynb)

- [paper](https://arxiv.org/pdf/2108.12409v2.pdf)
- [attention implementation](https://github.com/jaketae/alibi/blob/main/alibi/attention.py)

## Model

| Base Model | SFT Model     | RL Model       |
| ---------- | ------------- | -------------- |
| GPT,LlaMa  | Vicuna,alpaca | ChatGPT,Claude |

### Base Model

- [The Illustrated GPT-2 (Visualizing Transformer Language Models)](https://jalammar.github.io/illustrated-gpt2/)
- [nanoGPT](<(transformer/nanoGPT.md)>)
  - [github](https://github.com/karpathy/nanoGPT)
- GPT1
  - paper [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- GPT2
  - paper [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
  - [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
  - report [Release Strategies and the Social Impacts of Language Models](https://arxiv.org/pdf/1908.09203.pdf)
  - [release blog](https://openai.com/research/gpt-2-1-5b-release)
  - implementation code
    - [tf by openai](https://github.com/openai/gpt-2/blob/master/src/model.py)
    - [pytorch by huggingface](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py)
    - [by nanoGPT](https://github.com/karpathy/nanoGPT/blob/master/model.py)
    - [by cerebras](https://github.com/Cerebras/modelzoo/blob/main/modelzoo/transformers/pytorch/gpt2/gpt2_model.py)
  - GPT2 M
    - [hf](https://huggingface.co/gpt2-medium)
  - GPT2 L
    - [hf](https://huggingface.co/gpt2-large)
  - GPT2 XL
    - [hf](https://huggingface.co/gpt2-xl)
- GPT-J-6B
  - using [Mesh Transformer JAX](https://github.com/kingoflolz/mesh-transformer-jax/) trained on [the pile](https://pile.eleuther.ai/)
  - [hf](https://huggingface.co/EleutherAI/gpt-j-6b)
- GPT3
  - like GPT2 but use alternating dense and locally banded sparse attention patterns
  - paper [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf)
  - [github](https://github.com/openai/gpt-3)
- CodeX / code-davinci-002
  - GPT3 family on code
  - evaluation paper [Evaluating Large Language Models Trained on Code](https://arxiv.org/pdf/2107.03374.pdf)
- GPT4
  - [blog by openai](https://openai.com/research/gpt-4)
- [LlaMa](transformer/llama.md)
  - train on roughly 1.4T tokens from public data only
  - norm input at each layer; use SwiGLU; use RoPe
  - performance
    - LLaMa-13B matches GPT3-15B
    - LLaMa-65B matches Chinchilla-70B and PaLM-540B
  - [paper](https://arxiv.org/pdf/2302.13971.pdf)
  - [facebook github](https://github.com/facebookresearch/llama)
  - [download model slowly](https://github.com/shawwn/llama-dl)
  - implementation
    - [hf](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)
    - [facebook](https://github.com/facebookresearch/llama/blob/main/llama/model.py)
  - feedforward
    - [why](https://github.com/facebookresearch/llama/issues/245)
    - [GLU Variants Improve Transformer](https://arxiv.org/pdf/2002.05202.pdf)
  - inference
    - [llama.cpp](https://github.com/ggerganov/llama.cpp)
    - [llama2.c](https://github.com/karpathy/llama2.c)

### Supervised Fine-Tuning (SFT) Model

- Cerebras-GPT
  - gpt3 style arch with full attention, on the pile data, model size from 111M to 13B
  - [paper](https://arxiv.org/pdf/2304.03208.pdf)
  - [hf 13B](https://huggingface.co/cerebras/Cerebras-GPT-13B)
  - [code](https://github.com/Cerebras/modelzoo/blob/main/modelzoo/transformers/pytorch/gpt3/README.md) missing GPT3 implementation of banded sparse attention
- Vicuna-13B chatbot by LMSYS ORG
  - fine tune LLaMa with 70K user-shared conversations from ShareGPT
  - [blog](https://lmsys.org/blog/2023-03-30-vicuna/)
  - release delta weights on [llama](https://huggingface.co/docs/transformers/main/model_doc/llama)
  - [hf](https://huggingface.co/lmsys/vicuna-13b-delta-v1.1)
- [gpt4all](transformer/gpt4all.md)
  - fine tune gpt-j-6b with nomic-ai/gpt4all-j-prompt-generations with lora
  - [github](https://github.com/nomic-ai/gpt4all)
  - [technical report](https://static.nomic.ai/gpt4all/2023_GPT4All-J_Technical_Report_2.pdf)
- koala 13B by BAIR
  - LLaMa 13B with dialogue from open datasetma
  - "the key to building strong dialogue models may lie more in curating high-quality dialogue data that is diverse in user queries, rather than simply reformatting existing datasets as questions and answers."
  - [blog](https://bair.berkeley.edu/blog/2023/04/03/koala/)
- stanford alpaca
  - fine tune the original LLaMa 7B !!!
  - on 52k instruction (alpaca_data.json) generated from text-davinci-003
  - claimed to match the performance of text-davinci-003
  - [blog](https://crfm.stanford.edu/2023/03/13/alpaca.html)
  - [github](https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py)
  - [hf weight diff](https://huggingface.co/tatsu-lab/alpaca-7b-wdiff)

### RL Model

- ChatGPT from openai
  - [sharegpt](https://shareg.pt/4qj1DB0)
- Claude from Anthropic

## Technique

### Quantization

- GPT-J-6B-8bit
  - [8bit hf model](https://huggingface.co/hivemind/gpt-j-6B-8bit)
  - [tutorial](https://github.com/sleekmike/Finetune_GPT-J_6B_8-bit/blob/master/finetune_gpt_j_6B_8bit.ipynb)
  - [perplexity](https://nbviewer.org/urls/huggingface.co/hivemind/gpt-j-6B-8bit/raw/main/check_perplexity.ipynb)

### Safety

- [Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned](https://arxiv.org/pdf/2209.07858.pdf)

### Fine Tuning

- PEFT
  - Parameter-efficient fine-tuning of large-scale pre-trained language models
  - Towards a unified view of parameter-efficient transfer learning
  - [github](https://github.com/huggingface/peft)
  - LoRA
    - [paper](https://arxiv.org/pdf/2106.09685.pdf)
    - [github](https://github.com/microsoft/LoRA)
    - alpaca lora
      - [github](https://github.com/tloen/alpaca-lora)
      - [hf weights](https://huggingface.co/tloen/alpaca-lora-7b)
- instruction learning / zero shot
  - [awesome](https://github.com/RenzeLou/awesome-instruction-learning)
  - [Is Prompt All You Need? No. A Comprehensive and Broader View of Instruction Learning](https://arxiv.org/pdf/2303.10475.pdf)
  - SELF-INSTRUCT: Aligning Language Models with Self-Generated Instructions
    - grow instruction pair size with openai api
    - [paper](https://arxiv.org/pdf/2212.10560.pdf)
    - [github](https://github.com/yizhongw/self-instruct)
  - [LARGE LANGUAGE MODELS ARE HUMAN-LEVEL PROMPT ENGINEERS](https://arxiv.org/pdf/2211.01910.pdf)
    - using LLMs to generate and select instructions automatically, program synthesis

### Scaling / Emergent Behaviour

- [大语言模型的涌现能力：现象与解释](https://zhuanlan.zhihu.com/p/621438653)
- [paper Emergent Abilities of Large Language Models](https://arxiv.org/pdf/2206.07682.pdf)
- Chinchilla 7B
  - compute-optimal model, haiku on TPU
  - evaluate the trade off of model size and number of training tokens, given fixed flop budget
  - [paper](https://arxiv.org/pdf/2203.15556.pdf)
    - when model size doubled, training tokens should also be doubled

### Reasoning

- [REACT: SYNERGIZING REASONING AND ACTING IN LANGUAGE MODELS](https://arxiv.org/pdf/2210.03629.pdf)
  - combine CoT and Act
- [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/pdf/2305.10601.pdf)
  - a new framework for inference, genearlized over CoT to allow looking ahead and backtracking for decision making
  - [full paper review](https://www.youtube.com/watch?v=ut5kp56wW_4)

### Benchmark & Evaluation

- GPT-Fathom: Benchmarking Large Language Models to Decipher the Evolutionary Path towards {GPT-4} and Beyond
  - [paper](https://arxiv.org/pdf/2309.16583.pdf)
  - [github](https://github.com/GPT-Fathom/GPT-Fathom)
- [guidance](https://github.com/microsoft/guidance)
  - generation control

### Agent

- ProAgent: Building Proactive Cooperative AI with Large Language Models
  - [paper](https://arxiv.org/pdf/2308.11339.pdf)

### multi modal

- clip
  - [openai](https://github.com/openai/CLIP/blob/main/clip/clip.py)
  - [towhee](https://github.com/towhee-io/towhee/blob/1.1.3/towhee/models/clip/clip.py)
  - [hf openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32/tree/main)
  - [hf clip-ViT-B-32-multilingual](https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1/tree/main)
- clip4clip
  - [official](https://github.com/ArrowLuo/CLIP4Clip)
  - [towhee](https://github.com/towhee-io/towhee/blob/1.1.3/towhee/models/clip4clip/clip4clip.py#L119)
- owl-vit
  - [paper](https://arxiv.org/pdf/2205.06230.pdf)
  - [hf](https://huggingface.co/docs/transformers/main/en/model_doc/owlvit)

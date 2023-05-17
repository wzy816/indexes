# transformer

## AIGC

- [A Comprehensive Survey of AI-Generated Content (AIGC): A History of Generative AI from GAN to ChatGPT](https://arxiv.org/pdf/2303.04226.pdf)
- [Google "We Have No Moat, And Neither Does OpenAI"](https://www.semianalysis.com/p/google-we-have-no-moat-and-neither)
- A Survey of Large Language Model
  - [paper en](https://arxiv.org/pdf/2303.18223.pdf)
  - [paper cn](https://github.com/RUCAIBox/LLMSurvey/blob/main/assets/LLM_Survey_Chinese_0418.pdf)
  - [github](https://github.com/RUCAIBox/LLMSurvey)
- [Planning for AGI and beyond by Sam Altman](https://openai.com/blog/planning-for-agi-and-beyond)

## Attention Mechanism & Transformer

![](transformer/The-evolutionary-tree-of-modern-Large-Language-Models-LLMs-traces-the-development-of.png)

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- Attention Is All You Need :book:
- [Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)

## GPT

- [The Illustrated GPT-2 (Visualizing Transformer Language Models)](https://jalammar.github.io/illustrated-gpt2/)
- nanoGPT
  - [github](https://github.com/karpathy/nanoGPT)
  - [md](transformer/nanoGPT.md)
- GPT1
  - paper [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- GPT2
  - paper [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
  - [release blog](https://openai.com/research/gpt-2-1-5b-release)
  - implementation code
    - [tf by openai](https://github.com/openai/gpt-2/blob/master/src/model.py)
    - [pytorch by huggingface](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py)
    - [by nanoGPT](https://github.com/karpathy/nanoGPT/blob/master/model.py)
    - [by cerebras](https://github.com/Cerebras/modelzoo/tree/main/modelzoo/transformers/pytorch/gpt2)
  - [GPT2 M](https://huggingface.co/gpt2-medium)
  - [GPT2 L](https://huggingface.co/gpt2-large)
  - [GPT2 XL](https://huggingface.co/gpt2-xl))
- GPT-J-6B
  - using [Mesh Transformer JAX](https://github.com/kingoflolz/mesh-transformer-jax/) trained on [the pile](https://pile.eleuther.ai/)
  - [hf](https://huggingface.co/EleutherAI/gpt-j-6b)
- GPT3
  - paper [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf)
  - [github](https://github.com/openai/gpt-3)
- CodeX
  - GPT3 family on code
  - evaluation paper [Evaluating Large Language Models Trained on Code](https://arxiv.org/pdf/2107.03374.pdf)
- GPT4
  - [blog by openai](https://openai.com/research/gpt-4)
- ChatGPT
  - [share conversations](https://shareg.pt/4qj1DB0)
- LLaMa
  - [paper](https://arxiv.org/pdf/2302.13971.pdf)
  - [github](https://github.com/facebookresearch/llama)
  - [download model slowly](https://github.com/shawwn/llama-dl)

## public GPT

- Cerebras-GPT
  - gpt3 style arch with full attention, on the pile data, 111M to 13B
  - [paper](https://arxiv.org/pdf/2304.03208.pdf)
  - [hf 13B](https://huggingface.co/cerebras/Cerebras-GPT-13B)
  - [code](https://github.com/Cerebras/modelzoo/blob/main/modelzoo/transformers/pytorch/gpt3/README.md) missing GPT3 implementation of banded sparse attention
- Vicuna-13B chatbot by LMSYS ORG
  - fine tune LLaMa with 70K user-shared conversations from ShareGPT
  - [blog](https://lmsys.org/blog/2023-03-30-vicuna/)
  - release delta weights on [llama](https://huggingface.co/docs/transformers/main/model_doc/llama)
  - [hf](https://huggingface.co/lmsys/vicuna-13b-delta-v1.1)
- gpt4all
  - fine tune gpt-j-6b with nomic-ai/gpt4all-j-prompt-generations
  - [github](https://github.com/nomic-ai/gpt4all)
  - [md](transformer/gpt4all.md)
- koala 13B by BAIR
  - LLaMa 13B with dialogue from open dataset
  - "the key to building strong dialogue models may lie more in curating high-quality dialogue data that is diverse in user queries, rather than simply reformatting existing datasets as questions and answers."
  - [blog](https://bair.berkeley.edu/blog/2023/04/03/koala/)
- Alpaca 7B
  - LLaMa 7B on 52k instruction, performance sim. to text-davinci-003
  - [blog](https://crfm.stanford.edu/2023/03/13/alpaca.html)
  - [github](https://github.com/tatsu-lab/stanford_alpaca)

## quantization

- GPT-J-6B-8bit
  - [8bit hf model](https://huggingface.co/hivemind/gpt-j-6B-8bit)
  - [finetune 8bit](https://github.com/sleekmike/Finetune_GPT-J_6B_8-bit/blob/master/finetune_gpt_j_6B_8bit.ipynb)
  - [perplexity](https://nbviewer.org/urls/huggingface.co/hivemind/gpt-j-6B-8bit/raw/main/check_perplexity.ipynb)

## safety

- [Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned](https://arxiv.org/pdf/2209.07858.pdf)
- GPT2 report [Release Strategies and the Social Impacts of Language Models](https://arxiv.org/pdf/1908.09203.pdf)

## fine tuning

- PEFT
  - Parameter-efficient fine-tuning of large-scale pre-trained language models
  - Towards a unified view of parameter-efficient transfer learning
  - [peft](https://github.com/huggingface/peft)
  - LoRA
    - [paper](https://arxiv.org/pdf/2106.09685.pdf)
    - [github](https://github.com/microsoft/LoRA)
    - [alpaca-lora](https://github.com/tloen/alpaca-lora)
- instruction learning
  - [awesome](https://github.com/RenzeLou/awesome-instruction-learning)
  - [Is Prompt All You Need? No. A Comprehensive and Broader View of Instruction Learning](https://arxiv.org/pdf/2303.10475.pdf)

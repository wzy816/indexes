# gpt4all

```bash
# change env
export TRANSFORMERS_CACHE=/mnt/huggingface/hub
export HF_DATASETS_CACHE=/mnt/huggingface/datasets
export TRANSFORMERS_NO_ADVISORY_WARNINGS="true"

# install c++ 8
yum install centos-release-scl
yum install devtoolset-8-gcc devtoolset-8-gcc-c++
scl enable devtoolset-8 -- bash
which c++

# install git lfs
yum install git-lfs

# download dataset
git clone https://huggingface.co/datasets/nomic-ai/gpt4all-j-prompt-generations/
cd gpt4all-j-prompt-generations
git lfs pull

# clone repo
cd ..
git clone --recurse-submodules https://github.com/nomic-ai/gpt4all.git
cd gpt4all
git submodule update --init

conda create -n gpt4all python=3.9
conda activate gpt4fall

# modify finetune_gptj_lora yaml
# add dataset_version in train.py

accelerate launch --dynamo_backend=inductor --num_processes=1 --num_machines=1 --machine_rank=0 --deepspeed_multinode_launcher standard --mixed_precision=bf16  --use_deepspeed --deepspeed_config_file=configs/deepspeed/ds_config_gptj_lora.json train.py --config configs/train/finetune_gptj_lora_20230512.yaml

# trainable params: 3670016 || all params: 6054552800 || trainable%: 0.060615806339982696
# Len of train_dataloader: 768371
# Total training steps: 768871.0
```

## configs/train/finetune_gptj_lora_20230512.yaml

```yaml
# model/tokenizer
model_name: "EleutherAI/gpt-j-6B"
tokenizer_name: "EleutherAI/gpt-j-6B"
gradient_checkpointing: false
save_name: # CHANGE

# dataset
streaming: false
num_proc: 64 # for train_dataset.map in data.py
dataset_path: "nomic-ai/gpt4all-j-prompt-generations"
dataset_version: "v1.3-groovy"
max_length: 2048 # print from tokenizer used in data.py
batch_size: 1 # used in data.py

# train dynamics
lr: 2.0e-5
min_lr: 0
weight_decay: 0.0
eval_every: 5000
# eval_steps: 1050
save_every: 10000
log_grads_every: 5000
output_dir: "/mnt/gpt4all_output_gptj_lora_20230512/"
checkpoint: null
lora: true
warmup_steps: 500
num_epochs: 1

# logging
wandb: true
wandb_entity:
wandb_project_name: "gpt4all_gptj_lora_20230512"
seed: 42
```

# llama

[source code](https://github.com/wzy816/llama/tree/main)

```bash
conda actiavte llama
pip install transformers accelerate protobuf==3.20.3
git clone https://github.com/huggingface/transformers.git
cd transformers
python src/transformers/models/llama/convert_llama_weights_to_hf.py  --input_dir /mnt/llama/ --model_size 7B --output_dir /mnt/llama/hf_7B/

# see example.ipynb
```

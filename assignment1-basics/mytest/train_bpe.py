import os
import json
import random
from tokenizers import (
    Tokenizer, # 控制整个分词、解码、编码的过程，可以装配不同的组件
    decoders, # 将分词后的ids 转回原始文本
    models, # 包含不同的分词模型
    pre_tokenizers, # 文本预处理方式
    trainers # 训练纷纷那次模型的工具
)

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
# special_tokens = ["<|endoftext|>"]
special_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]

trainer = trainers.BpeTrainer(
    vocab_size=6400,
    special_tokens=special_tokens,
    show_progress=True,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
)

# data_path = "/home/spsong/Code/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
data_path = "/home/spsong/Code/minimind/dataset/pretrain_hq.jsonl"

def read_texts(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            yield line

def read_texts_from_jsonl(data_path):
    lens = 100
    with open(data_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx > lens:
                break
            data = json.loads(line)
            yield data["text"]

# text_gen = read_texts(data_path)
text_gen = read_texts_from_jsonl(data_path)

tokenizer.train_from_iterator(text_gen, trainer=trainer)

tokenizer_dir = "/home/spsong/Code/cs336/assignment1-basics/mytest/minimind_model"
os.makedirs(tokenizer_dir, exist_ok=True)
tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
tokenizer.model.save(tokenizer_dir)

config = {
    "add_bos_token": False,
    "add_eos_token": False,
    "add_prefix_space": False,
    "added_tokens_decoder": {
        "0": {
            "content": "<|endoftext|>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True
        },
        "1": {
            "content": "<|im_start|>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True
        },
        "2": {
            "content": "<|im_end|>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True
        }
    },
    "additional_special_tokens": [],
    "bos_token": "<|im_start|>",
    "clean_up_tokenization_spaces": False,
    "eos_token": "<|im_end|>",
    "legacy": True,
    "model_max_length": 32768,
    "pad_token": "<|endoftext|>",
    "sp_model_kwargs": {},
    "spaces_between_special_tokens": False,
    "tokenizer_class": "PreTrainedTokenizerFast",
    "unk_token": "<|endoftext|>",
    "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{{ '<|im_start|>system\\n' + system_message + '<|im_end|>\\n' }}{% else %}{{ '<|im_start|>system\\nYou are a helpful assistant<|im_end|>\\n' }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|im_start|>user\\n' + content + '<|im_end|>\\n<|im_start|>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<|im_end|>' + '\\n' }}{% endif %}{% endfor %}"
}


with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
    json.dump(config, config_file, ensure_ascii=False, indent=4)

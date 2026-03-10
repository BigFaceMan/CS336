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

def eval_tokenizer():
    from transformers import AutoTokenizer

    # 加载预训练的tokenizer
    tokenizer = AutoTokenizer.from_pretrained("/home/spsong/Code/minimind/model")

    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": '你来自哪里？'},
        {"role": "assistant", "content": '我来自地球'}
    ]
    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
    

    # 获取实际词汇表长度（包括特殊符号）
    actual_vocab_size = len(tokenizer)
    print('tokenizer实际词表长度：', actual_vocab_size)

    model_inputs = tokenizer(new_prompt)
    
    print('encoder长度：', len(model_inputs['input_ids']))

    input_ids = model_inputs['input_ids']
    response = tokenizer.decode(input_ids, skip_special_tokens=False)
    print('decoder和原始文本是否一致：', response == new_prompt)

    print('\n输入文本：\n',new_prompt,'\n')   
    print('解码文本：\n',response,'\n')  

eval_tokenizer()
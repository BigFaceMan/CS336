import regex as re
import cProfile
from tools.profile import profile
from tqdm import tqdm

def log(name, val):
    return 
    # print(f"{name} is {val}, len is {len(val)}")


def get_docs(input_path, special_tokens):
    with open(input_path, 'r') as f:
        content = f.read()

    if not special_tokens:
        docs = [content]
    else:
        pattern = "|".join(map(re.escape, special_tokens))
        log("pattern", pattern)
        docs = [d for d in re.split(pattern, content) if d]

    return docs

def pre_tokenization(docs: list[str]):
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_token_cache = dict()
    for doc in docs:
        pattern_items = re.findall(pattern, doc)
        for item in pattern_items:
            item_utf8 = item.encode('utf-8')
            pre_token_cache[tuple(item_utf8)] = pre_token_cache.get(tuple(item_utf8), 0) + 1
    return pre_token_cache

def get_max_frequency_pair(pre_token_cache: dict[tuple, int], vocab: dict[int, bytes]):
    cnt = {} 
    for key, val in pre_token_cache.items():
        for i in range(len(key) - 1):
            cnt[tuple([key[i], key[i + 1]])] = cnt.get(tuple([key[i], key[i + 1]]), 0) + val
    
    key_mx, val_mx = tuple(), 0
    key_mx_ls = [] 

    for key, val in cnt.items():
        if val > val_mx:
            key_mx = key
            val_mx = val
        elif val == val_mx:
            ord1 = tuple([vocab[key[0]], vocab[key[1]]])
            ord2 = tuple([vocab[key_mx[0]], vocab[key_mx[1]]])
            if ord1 > ord2:
                key_mx = key
                val_mx = val

    return key_mx, val_mx


def train_bpe(input_path: str, 
            vocab_size: int, 
            special_tokens: list[str]):
    vocab = dict()
    merge = list()

    docs = get_docs(input_path, special_tokens)

    pre_token_cache = pre_tokenization(docs)

    log("pre_token_cache", pre_token_cache)

    vocab_idx = 0

    for i in range(256):
        vocab[vocab_idx] = bytes([i])
        vocab_idx += 1

    for special_token in special_tokens:
        vocab[vocab_idx] = special_token.encode("utf-8")
        vocab_idx += 1

    process_bar = tqdm(total=vocab_size, initial=vocab_idx)
    flag = True

    while vocab_idx < vocab_size:
        key_mx, val_mx = get_max_frequency_pair(pre_token_cache, vocab)

        merge.append(tuple([vocab[key_mx[0]], vocab[key_mx[1]]]))

        vocab[vocab_idx] = vocab[key_mx[0]] + vocab[key_mx[1]]

        pre_token_cache_new = {}

        for key, val in pre_token_cache.items():
            i = 0
            key_new = list()
            while i < len(key):
                if i + 1 < len(key) and key[i] == key_mx[0] and key[i + 1] == key_mx[1]:
                    key_new.append(vocab_idx)
                    i += 2
                else:
                    key_new.append(key[i])
                    i += 1
            
            pre_token_cache_new[tuple(key_new)] = val
        pre_token_cache = pre_token_cache_new
        vocab_idx += 1
        process_bar.update(1)
    process_bar.close()
    return vocab, merge
    

if __name__ == "__main__":
    input_path = "/home/spsong/Code/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 500
    special_token = ["<|endoftext|>"]

    vocab, merge = train_bpe(input_path, vocab_size, special_token)
    

import os
import regex as re
import cProfile
import pstats
import multiprocessing
from typing import BinaryIO
from tqdm import tqdm
from multiprocessing import Process, Queue
from collections import Counter

def log(name, val):
    return 
    # print(f"{name} is {val}, len is {len(val)}")



def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pre_tokenization(docs: list[str]):
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    pre_token_cache = dict()
    for doc in docs:
        pattern_items = re.findall(pattern, doc)
        for item in pattern_items:
            item_utf8 = item.encode('utf-8')
            pre_token_cache[tuple(item_utf8)] = pre_token_cache.get(tuple(item_utf8), 0) + 1

    return pre_token_cache

def worker(docs, q):
    process_name = multiprocessing.current_process().name
    print(f"当前进程: {process_name}")

    pre_token_cache = pre_tokenization(docs)
    q.put(pre_token_cache)
    
def get_pre_token(input_path, special_tokens):
    with open(input_path, 'r') as f:
        content = f.read()

    if not special_tokens:
        docs = [content]
    else:
        pattern = "|".join(map(re.escape, special_tokens))
        log("pattern", pattern)
        docs = [d for d in re.split(pattern, content) if d]
    pre_toekn_cache = pre_tokenization(docs)

    return pre_toekn_cache
    
        

    

def get_doc(input_path, special_tokens, num_works):
    q = Queue()
    processes = []
    ## Usage
    with open(input_path, "rb") as f:
        num_processes = num_works
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            if not special_tokens:
                docs = [chunk]
            else:
                pattern = "|".join(map(re.escape, special_tokens))
                log("pattern", pattern)
                docs = [d for d in re.split(pattern, chunk) if d]

            p = Process(target=worker, args=(docs, q))
            p.start()
            processes.append(p)
        
        result = []

        for _ in processes:
            result.append(q.get())

        for p in processes:
            p.join()
            

    pre_token_caches = {}

    for d in result:
        for k, v in d.items():
            pre_token_caches[k] = pre_token_caches.get(k, 0) + v
            
    return pre_token_caches

def build_reverse_index(pre_token_cache: dict[tuple, int]):
    pre_ids_count = []
    pair_count = Counter()
    pair_index = dict()

    for k, v in pre_token_cache.items():
        pre_ids_count.append({"ids" : k, "cnt" : v })
        for i in range(len(k) - 1):
            pair_key = tuple([k[i], k[i + 1]])
            pair_count[pair_key] = pair_count.get(pair_key, 0) + v
            pair_index.setdefault(pair_key, set()).add(len(pre_ids_count) - 1)
        
    return pre_ids_count, pair_count, pair_index


def run_merge(pre_ids_count: list[dict], pair_count: dict[tuple, int], pair_index: dict[tuple[int, int], set], vocab_size: int, vocab: dict[int, bytes]):
    merges = []
    vocab_idx = len(vocab)

    process_bar = tqdm(total=vocab_size, initial=vocab_idx)

    while vocab_idx < vocab_size:

        if not pair_count:
            break 

        pair_mx = max(pair_count, key=lambda p:(pair_count[p], vocab[p[0]], vocab[p[1]]))

        vocab[vocab_idx] = vocab[pair_mx[0]] + vocab[pair_mx[1]]
        merges.append(tuple([vocab[pair_mx[0]], vocab[pair_mx[1]]]))

        index_2_update = pair_index.get(pair_mx, set()).copy()

        for index in index_2_update:
            item = pre_ids_count[index]
            ids = item["ids"]
            cnt = item["cnt"]
            i = 0
            new_ids = []

            while i < len(ids):
                if i + 1 < len(ids) and ids[i] == pair_mx[0] and ids[i + 1] == pair_mx[1]:
                    new_ids.append(vocab_idx)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            
            old_pair = [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]           
            new_pair = [(new_ids[i], new_ids[i + 1]) for i in range(len(new_ids) - 1)]           

            for p in old_pair:
                pair_count[p] -= cnt
                if not pair_count[p]:
                    pair_count.pop(p, None)
                if p in pair_index:
                    pair_index[p].discard(index)
                    if not pair_index[p]:
                        pair_index.pop(p, None)
            
            for p in new_pair:
                pair_count[p] = pair_count.get(p, 0) + cnt
                pair_index.setdefault(p, set()).add(index)
            
            item["ids"] = new_ids
            
        vocab_idx += 1
        process_bar.update(1)

    process_bar.close()
            

    return vocab, merges
    


def train_bpe(input_path: str, 
            vocab_size: int, 
            special_tokens: list[str],
            num_works: int):
    vocab = dict()
    pre_token_cache = get_doc(input_path, special_tokens, num_works)
    # pre_token_cache = get_pre_token(input_path, special_tokens)

    vocab_idx = 0

    for i in range(256):
        vocab[vocab_idx] = bytes([i])
        vocab_idx += 1

    for special_token in special_tokens:
        vocab[vocab_idx] = special_token.encode("utf-8")
        vocab_idx += 1
    
    pre_ids_count, pair_count, pair_index = build_reverse_index(pre_token_cache)

    vocab, merges = run_merge(pre_ids_count, pair_count, pair_index, vocab_size, vocab)

    return vocab, merges
    

if __name__ == "__main__":
    input_path = "/home/spsong/Code/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
    # input_path = "/home/spsong/Code/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 500
    special_token = ["<|endoftext|>"]

    profiler = cProfile.Profile()
    profiler.enable()

    vocab, merge = train_bpe(input_path, vocab_size, special_token, 4)

    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats("cumtime")  # 按累计时间排序
    stats.print_stats(20)  
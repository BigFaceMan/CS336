import os
import regex as re
from typing import BinaryIO
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field


PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def _pre_tokenize_chunk(args: tuple) -> dict[tuple[int, ...], int]:
    chunk, special_tokens = args
    if not special_tokens:
        docs = [chunk]
    else:
        pattern = "|".join(map(re.escape, special_tokens))
        docs = [d for d in re.split(pattern, chunk) if d]

    pre_token_cache: dict[tuple[int, ...], int] = {}
    for doc in docs:
        items = PATTERN.findall(doc)
        for item in items:
            item_utf8 = item.encode("utf-8")
            key = tuple(item_utf8)
            pre_token_cache[key] = pre_token_cache.get(key, 0) + 1

    return pre_token_cache


@dataclass
class PreTokenizer:
    special_tokens: list[str] = field(default_factory=list)
    split_special_token: bytes = field(default=b"<|endoftext|>")

    def _split_by_special_tokens(self, content: str) -> list[str]:
        if not self.special_tokens:
            return [content]
        pattern = "|".join(map(re.escape, self.special_tokens))
        return [d for d in re.split(pattern, content) if d]

    def tokenize(self, text: str) -> list[tuple[int, ...]]:
        docs = self._split_by_special_tokens(text)
        result = []
        for doc in docs:
            items = PATTERN.findall(doc)
            for item in items:
                item_utf8 = item.encode("utf-8")
                result.append(tuple(item_utf8))
        return result

    def tokenize_file(self, input_path: str, num_workers: int = 1) -> dict[tuple[int, ...], int]:
        if num_workers <= 1:
            return self._tokenize_file_serial(input_path)
        return self._tokenize_file_parallel(input_path, num_workers)

    def _tokenize_file_serial(self, input_path: str) -> dict[tuple[int, ...], int]:
        with open(input_path, "r", encoding="utf-8") as f:
            content = f.read()

        docs = self._split_by_special_tokens(content)
        pre_token_cache: dict[tuple[int, ...], int] = {}

        for doc in docs:
            items = PATTERN.findall(doc)
            for item in items:
                item_utf8 = item.encode("utf-8")
                key = tuple(item_utf8)
                pre_token_cache[key] = pre_token_cache.get(key, 0) + 1

        return pre_token_cache

    def _tokenize_file_parallel(self, input_path: str, num_workers: int) -> dict[tuple[int, ...], int]:
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_workers, self.split_special_token)

        chunks = []
        with open(input_path, "rb") as f:
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                chunks.append((chunk, self.special_tokens))

        pre_token_caches: dict[tuple[int, ...], int] = {}

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_pre_tokenize_chunk, args) for args in chunks]
            for future in as_completed(futures):
                result = future.result()
                for k, v in result.items():
                    pre_token_caches[k] = pre_token_caches.get(k, 0) + v

        return pre_token_caches


@dataclass
class BPETrainer:
    special_tokens: list[str] = field(default_factory=list)
    vocab: dict[int, bytes] = field(default_factory=dict)
    merges: list[tuple[bytes, bytes]] = field(default_factory=list)
    pre_tokenizer: PreTokenizer = field(init=False)

    def __post_init__(self):
        self.pre_tokenizer = PreTokenizer(special_tokens=self.special_tokens)
        self.vocab = {i: bytes([i]) for i in range(256)}
        self._vocab_idx = len(self.vocab)

        for special_token in self.special_tokens:
            self.vocab[self._vocab_idx] = special_token.encode("utf-8")
            self._vocab_idx += 1

    @property
    def vocab_size_with_special(self) -> int:
        return self._vocab_idx

    def _add_merged_token(self, pair: tuple[int, int]) -> int:
        merged_bytes = self.vocab[pair[0]] + self.vocab[pair[1]]
        self.merges.append((self.vocab[pair[0]], self.vocab[pair[1]]))
        self.vocab[self._vocab_idx] = merged_bytes
        self._vocab_idx += 1
        return self._vocab_idx - 1

    def _build_indices(self, pre_token_cache: dict[tuple[int, ...], int]):
        pre_ids_count: list[tuple[tuple[int, ...], int]] = []
        pair_count: dict[tuple[int, int], int] = {}
        pair_index: dict[tuple[int, int], list[int]] = {}

        for token_bytes, count in pre_token_cache.items():
            pre_ids_count.append((token_bytes, count))
            idx = len(pre_ids_count) - 1

            for i in range(len(token_bytes) - 1):
                pair = (token_bytes[i], token_bytes[i + 1])
                pair_count[pair] = pair_count.get(pair, 0) + count
                if pair not in pair_index:
                    pair_index[pair] = []
                pair_index[pair].append(idx)

        return pre_ids_count, pair_count, pair_index

    def _find_best_pair(self, pair_count: dict[tuple[int, int], int]) -> tuple[int, int] | None:
        if not pair_count:
            return None
        return max(pair_count, key=lambda p: (pair_count[p], self.vocab[p[0]], self.vocab[p[1]]))

    def train(self, input_path: str, target_vocab_size: int, num_workers: int = 4):
        pre_token_cache = self.pre_tokenizer.tokenize_file(input_path, num_workers)

        pre_ids_count, pair_count, pair_index = self._build_indices(pre_token_cache)

        target_size = target_vocab_size

        with tqdm(total=target_size, initial=self._vocab_idx, desc="bpe_merge") as pbar:
            while self._vocab_idx < target_size:
                if not pair_count:
                    break

                best_pair = self._find_best_pair(pair_count)
                if best_pair is None:
                    break

                new_token_id = self._add_merged_token(best_pair)
                indices_to_update = pair_index.pop(best_pair, [])

                for idx in indices_to_update:
                    token_ids, cnt = pre_ids_count[idx]
                    new_ids = []

                    i = 0
                    while i < len(token_ids):
                        if i + 1 < len(token_ids) and token_ids[i] == best_pair[0] and token_ids[i + 1] == best_pair[1]:
                            new_ids.append(new_token_id)
                            i += 2
                        else:
                            new_ids.append(token_ids[i])
                            i += 1

                    old_pairs = [(token_ids[i], token_ids[i + 1]) for i in range(len(token_ids) - 1)]
                    new_pairs = [(new_ids[i], new_ids[i + 1]) for i in range(len(new_ids) - 1)]

                    for p in old_pairs:
                        pair_count[p] -= cnt
                        if pair_count[p] == 0:
                            del pair_count[p]
                            if p in pair_index:
                                del pair_index[p]

                    for p in new_pairs:
                        pair_count[p] = pair_count.get(p, 0) + cnt
                        if p not in pair_index:
                            pair_index[p] = []
                        pair_index[p].append(idx)

                    pre_ids_count[idx] = (tuple(new_ids), cnt)

                pbar.update(1)

        return self.vocab, self.merges
    
    def save_vocab(self, path: str):
        import json

        vocab_json = {str(k): list(v) for k, v in self.vocab.items()}
        with open(path, "w") as f:
            json.dump(vocab_json, f)


    def save_merges(self, path: str):
        with open(path, "w") as f:
            for merge in self.merges:
                f.write(f"{merge[0].decode('utf-8', errors='replace')} {merge[1].decode('utf-8', errors='replace')}\n")


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    num_workers: int = 4,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if special_tokens is None:
        special_tokens = []

    trainer = BPETrainer(special_tokens=special_tokens)
    return trainer.train(input_path, vocab_size, num_workers)




if __name__ == "__main__":
    input_path = "/home/spsong/Code/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 500
    num_workers = 4
    special_tokens = ["<|endoftext|>"]
    output_dir = "/home/spsong/Code/cs336/assignment1-basics/data"

    # vocab, merges = train_bpe(input_path, vocab_size, special_token, 4)

    trainer: BPETrainer = BPETrainer(special_tokens=special_tokens)
    vocab, merges = trainer.train(input_path, vocab_size, num_workers)

    print(f"Final vocab size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")

    trainer.save_vocab(f"{output_dir}/vocab.json")
    trainer.save_merges(f"{output_dir}/merges.txt")
    print(f"Saved vocab to {output_dir}/vocab.json")
    print(f"Saved merges to {output_dir}/merges.txt")

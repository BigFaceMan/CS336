from collections.abc import Iterable
from collections.abc import Iterator
from dataclasses import field
import regex as re

PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


class Tokenizer:
    vocab: dict[int, bytes] = field(default_factory=dict)
    merges: list[tuple[bytes, bytes]] = field(default_factory=list)
    special_tokens: list[str] = field(default_factory=list)
    bytes_2_id: dict[bytes, int]

    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        # update vocab
        exits_tokens = set(self.vocab.values())
        next_id = max(self.vocab.keys() if self.vocab else -1) + 1
        if special_tokens:
            for special_token in special_tokens:
                special_token_bytes = special_token.encode("utf-8")
                if special_token_bytes not in exits_tokens:
                    exits_tokens.add(special_token_bytes)
                    vocab[next_id] = special_token_bytes
                    next_id += 1

        self.bytes_2_id = {v: k for k, v in self.vocab.items()}

        self.bpe_ranks = {pair : i for i, pair in enumerate(self.merges)}

        self.gpt2_regex = re.compile(PATTERN)
        
        self.cache = {}


    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens=None):
        """
        vocab_json
            type is dict
            k str(k)
            v base64 encoded bytes

        merges_json
            type is list
            tuple, base64 encoded bytes

        """
        import json
        import base64

        with open(vocab_filepath, "r") as f:
            vocab_json: dict = json.load(f)

        with open(merges_filepath, "r") as f:
            merges_json: list = json.load(f)

        vocab = {int(k): base64.b64decode(v) for k, v in vocab_json.items()}
        merges = [tuple([base64.b64decode(v[0]), base64.b64decode(v[1])]) for v in merges_json]

        return Tokenizer(vocab, merges, special_tokens)

    # def _bpe_encode(self, txi: str):
    #     # 如果用这样就是假设 vocab (0, 255) 是正常初始化的
    #     txi_ids = list(txi.encode("utf-8"))
    #     for a, b in self.merges:
    #         i = 0
    #         new_txi_ids = []
    #         while i < len(txi_ids):
    #             if i + 1 < len(txi_ids) and self.vocab[txi_ids[i]] == a and self.vocab[txi_ids[i + 1]] == b:
    #                 new_bytes = self.vocab[txi_ids[i]] + self.vocab[txi_ids[i + 1]]
    #                 new_id = self.bytes_2_id[new_bytes]
    #                 new_txi_ids.append(new_id)
    #                 i += 2
    #             else:
    #                 new_txi_ids.append(txi_ids[i])
    #                 i += 1
    #         txi_ids = new_txi_ids
    #     return txi_ids

    def _bpe_encode(self, txi: str) -> list[int]:
        """
        txi: pre_token后的字符串
        """
        token_bytes = txi.encode("utf-8")
        if token_bytes in self.cache:
            return self.cache[token_bytes]

        tokens_id = [self.bytes_2_id[bytes([i])] for i in token_bytes]

        while len(tokens_id) > 1:
            rank_mn = float('inf')
            merge_pair = None 
            merge_index = -1
            i = 0
            while i < len(tokens_id):
                if i + 1 < len(tokens_id):
                    pairi = (self.vocab[tokens_id[i]], self.vocab[tokens_id[i + 1]])
                    if pairi in self.bpe_ranks:
                        rank = self.bpe_ranks[pairi]
                        if rank < rank_mn:
                            rank_mn = rank
                            merge_pair = pairi
                            merge_index = i
                i += 1
            if merge_pair is None:
                break
        
            new_token_bytes = merge_pair[0] + merge_pair[1]
            new_token_id = self.bytes_2_id[new_token_bytes]
        
            new_tokens_id = []
            i = 0
            while i < len(tokens_id):
                if i + 1 < len(tokens_id):
                    pairi = (self.vocab[tokens_id[i]], self.vocab[tokens_id[i + 1]])
                    if pairi == merge_pair:
                        new_tokens_id.append(new_token_id)
                        i += 2
                        continue
                new_tokens_id.append(tokens_id[i])
                i += 1
            tokens_id = new_tokens_id
        
        self.cache[token_bytes] = tokens_id
        
        return tokens_id

    def encode(self, text: str) -> list[int]:
        if not self.special_tokens:
            docs = [text]
        else:
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            pattern = f"({'|'.join(map(re.escape, sorted_tokens))})"
            docs = re.split(pattern, text)

        encode_ids = []
        for doci in docs:
            if not doci:
                continue

            if doci in self.special_tokens:
                encode_ids.append(self.bytes_2_id[doci.encode("utf-8")])
            else:
                pre_tokens = self.gpt2_regex.findall(doci)

                for txi in pre_tokens:
                    txi_ids = self._bpe_encode(txi)
                    encode_ids.extend(txi_ids)

        return encode_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text_chunk in iterable:
            yield from self.encode(text_chunk)

    def decode(self, ids: list[int]) -> str:
        decode_bytes = bytes()
        for idi in ids:
            if idi in self.vocab:
                decode_bytes += self.vocab[idi]
        return decode_bytes.decode("utf-8", errors="replace")


if __name__ == "__main__":
    print("------------------------------------TEST1------------------------------------")
    vocab_filepath = "/home/spsong/Code/cs336/assignment1-basics/output/tokenizer_small_story/vocab.json"
    merges_filepath = "/home/spsong/Code/cs336/assignment1-basics/output/tokenizer_small_story/merges.json"
    special_tokens = ["<|endoftext|>", "aaaaa"]
    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)
    str = "a<|endoftext|>"
    encode_ids = tokenizer.encode(str)
    decode_str = tokenizer.decode(encode_ids)
    print(f"str : {str}")
    print(f"decode_str : {decode_str}")
    assert str == decode_str

    print("------------------------------------TEST2------------------------------------")
    # test iterable
    all_ids = []
    with open("/home/spsong/Code/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt") as f:
        for _id in tokenizer.encode_iterable(f):
            # print(f"get id {_id}")
            all_ids.append(_id)

    with open("/home/spsong/Code/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt") as f:
        target_content = f.read()

    assert tokenizer.decode(all_ids) == target_content

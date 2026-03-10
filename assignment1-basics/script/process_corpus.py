import argparse
import os
import yaml
import time
from cs336_basics.tokenizer import Tokenizer


def get_base_path():
    config_dir = os.path.join(os.path.dirname(__file__), "..", "config")
    base_yaml_path = os.path.join(config_dir, "base.yaml")
    with open(base_yaml_path) as f:
        config = yaml.safe_load(f)
    return config.get("base_path", ".")


def main(input_path, vocab_path, output_bin, chunk_size_mb):
    BASE_PATH = get_base_path()

    vocab_filepath = os.path.join(BASE_PATH, vocab_path, "vocab.json")
    merges_filepath = os.path.join(BASE_PATH, vocab_path, "merges.json")
    input_file = os.path.join(BASE_PATH, input_path)
    output_file = os.path.join(BASE_PATH, output_bin)

    print(f"Loading tokenizer from {vocab_filepath} and {merges_filepath}")
    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, ["<|endoftext|>"])

    print(f"Processing {input_file} -> {output_file}")
    tokenizer.process_corpus(input_file, output_file, chunk_size_mb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process corpus to binary tokens")

    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--vocab_path", type=str, default=None)
    parser.add_argument("--output_bin", type=str, default=None)
    parser.add_argument("--chunk_size_mb", type=int, default=None)

    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    start = time.time()
    main(
        input_path=args.input_path or config.get("input_path"),
        vocab_path=args.vocab_path or config.get("vocab_path"),
        output_bin=args.output_bin or config.get("output_bin"),
        chunk_size_mb=args.chunk_size_mb or config.get("chunk_size_mb", 50),
    )
    end = time.time()
    print(f"Cost time: {end - start:.2f}s")

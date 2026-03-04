import argparse
import os
import yaml
import time
from cs336_basics.bpe_optimized import BPETrainer


def main(input_path, vocab_size, num_workers, special_tokens, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 初始化训练器
    trainer = BPETrainer(special_tokens=special_tokens)

    # 训练
    vocab, merges = trainer.train(input_path, vocab_size, num_workers)

    print(f"Final vocab size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")

    # 保存结果
    trainer.save_vocab(f"{output_dir}/vocab.json")
    trainer.save_merges(f"{output_dir}/merges.json")
    print(f"Saved vocab to {output_dir}/vocab.json")
    print(f"Saved merges to {output_dir}/merges.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer")

    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--input_path", type=str, default=None, help="Path to input text file")
    parser.add_argument("--vocab_size", type=int, default=None, help="Target vocabulary size")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel workers")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save vocab and merges")
    parser.add_argument("--special_tokens", type=str, nargs="*", default=None, help="List of special tokens")

    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    start = time.time()
    main(
        input_path=args.input_path or config.get("input_path"),
        vocab_size=args.vocab_size or config.get("vocab_size", 500),
        num_workers=args.num_workers or config.get("num_workers", 4),
        special_tokens=args.special_tokens or config.get("special_tokens", ["<|endoftext|>"]),
        output_dir=args.output_dir or config.get("output_dir"),
    )
    end = time.time()
    print(f"cost time {end - start}")

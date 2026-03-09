import os
import time
import argparse
import yaml
import torch
from dataclasses import dataclass, field, asdict
from typing import Optional
import swanlab
import numpy as np
from tqdm import tqdm
from cs336_basics.module import TransformerLM
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.utils import get_batch, cross_entropy, save_checkpoint, load_checkpoint
from cs336_basics.optimizer import AdamW, CosineAnnealingSchedue, grad_clip


@dataclass
class TrainConfig:
    begin_iters: int = 0
    end_iters: int = 10000
    save_interval: int = 10000
    lr: float = 0.001
    batch_size: int = 32
    max_seq_len: int = 1000
    num_layers: int = 6
    num_heads: int = 8
    d_model: int = 512
    d_ff: int = 2048
    device: str = "cuda"
    seed: int = 42
    data_path: Optional[str] = None
    vocab_path: Optional[str] = None
    output_path: Optional[str] = None
    theta: int = 10000
    vocab_size: int = 10000
    config: Optional[str] = field(default=None, repr=False)


def load_dataset(config: TrainConfig):
    # 1. dataload
    assert config.vocab_path is not None, "vocab_path is required"
    tokenizer = Tokenizer.from_files(
        os.path.join(config.vocab_path, "vocab.json"), os.path.join(config.vocab_path, "merges.json")
    )

    dataset = []

    with open(config.data_path, 'r') as f:
        for ids in tokenizer.encode_iterable(f):
            dataset.append(ids)

    return dataset, tokenizer


def train(config: TrainConfig):
    # load dataset
    dataset, tokenizer = load_dataset(config)
    dataset = np.array(dataset)

    print(type(dataset))
    print(f"dataset len {len(dataset)}")

    swanlab.init(
        project="cs336_lab1_small_story", 
        config= asdict(config)
    )

    print(f"Config {config}")

    # init model optimizer schedule

    model = TransformerLM(
        config.d_model,
        config.num_heads,
        config.num_layers,
        config.d_ff,
        config.max_seq_len,
        config.theta,
        config.vocab_size,
        config.device,
        torch.float32,
    ).to(config.device)

    optimizer = AdamW(model.parameters(), config.lr, 0.01, (0.9, 0.999))
    # schedule = CosineAnnealingSchedue(optimizer, config.lr, )


    process_bar = tqdm(range(config.begin_iters, config.end_iters), desc="llm train")

    # train loop
    model.train()
    for iter in range(config.begin_iters, config.end_iters):
        inputs, targets = get_batch(dataset, config.batch_size, config.max_seq_len, config.device)

        outputs = model(inputs)

        loss = cross_entropy(outputs, targets)

        loss.backward()

        # grad_clip(model.parameters(), 1.0)
        

        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            if iter % 10 == 0:
                swanlab.log({"loss" : loss.item()}, iter)
                process_bar.set_postfix({"loss" : loss.item()})
                process_bar.update(10)
            if iter == config.end_iters - 1:
                process_bar.close()

            if iter % config.save_interval == 0:
                save_checkpoint(model, optimizer, iter, os.path.join(config.output_path, "checkpoint.pth"))

    save_checkpoint(model, optimizer, iter, os.path.join(config.output_path, "checkpoint.pth"))




if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--config", type=str, default=None, help="Path to yaml config file")
    parse.add_argument("--begin_iters", type=int, default=0)
    parse.add_argument("--end_iters", type=int, default=10000)
    parse.add_argument("--save_interval", type=int, default=100)
    parse.add_argument("--data_path", type=str)
    parse.add_argument("--vocab_path", type=str)
    parse.add_argument("--output_path", type=str)
    parse.add_argument("--lr", type=float, default=None)
    parse.add_argument("--batch_size", type=int, default=None)
    parse.add_argument("--max_seq_len", type=int, default=None)
    parse.add_argument("--num_layers", type=int, default=None)
    parse.add_argument("--num_heads", type=int, default=None)
    parse.add_argument("--d_model", type=int, default=None)
    parse.add_argument("--d_ff", type=int, default=None)
    parse.add_argument("--theta", type=int, default=None)
    parse.add_argument("--vocab_size", type=int, default=None)
    parse.add_argument("--device", type=str, default=None)
    parse.add_argument("--seed", type=int, default=None)

    cli_args = parse.parse_args()

    yaml_config = {}
    if cli_args.config:
        with open(cli_args.config, "r") as f:
            yaml_config = yaml.safe_load(f) or {}

    default_config = TrainConfig()
    default_dict = {k: v for k, v in vars(default_config).items() if v is not None}

    merged = {**default_dict, **yaml_config}
    merged = {k: v for k, v in merged.items() if v is not None}

    parse.set_defaults(**merged)

    args = parse.parse_args()

    config = TrainConfig(**vars(args))

    os.makedirs(config.output_path, exist_ok=True)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    train(config)

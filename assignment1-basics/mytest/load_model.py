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
from cs336_basics.utils import get_batch, cross_entropy, save_checkpoint, load_checkpoint
from cs336_basics.optimizer import AdamW, CosineAnnealingSchedue, grad_clip


@dataclass
class TrainConfig:
    begin_iters: int = 0
    end_iters: int = 10000
    save_interval: int = 10000
    lr: float = 0.001
    max_norm: float = 1.0
    min_lr: float = 6e-5
    batch_size: int = 32
    max_seq_len: int = 1000
    num_layers: int = 6
    num_heads: int = 8
    d_model: int = 512
    d_ff: int = 2048
    device: str = "cuda"
    seed: int = 42
    train_data_path: Optional[str] = None
    val_data_path: Optional[str] = None
    vocab_path: Optional[str] = None
    output_path: Optional[str] = None
    theta: int = 10000
    vocab_size: int = 10000
    warmup_iter: int = 1000
    config: Optional[str] = field(default=None, repr=False)
    exp_name: Optional[str] = None


def train(config: TrainConfig):
    model = TransformerLM(
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        theta=config.theta,
        vocab_size=config.vocab_size,
        device=config.device,
        dtype=torch.float32,
    ).to(config.device)
    print("Model is :")
    print(model)





if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    # path
    parse.add_argument("--config", type=str, default=None, help="Path to yaml config file")
    parse.add_argument("--begin_iters", type=int, default=0)
    parse.add_argument("--end_iters", type=int, default=10000)
    parse.add_argument("--save_interval", type=int, default=100)
    parse.add_argument("--train_data_path", type=str)
    parse.add_argument("--val_data_path", type=str)
    parse.add_argument("--output_path", type=str)

    # model params
    parse.add_argument("--vocab_path", type=str)
    parse.add_argument("--vocab_size", type=int, default=1000)
    parse.add_argument("--batch_size", type=int, default=32)
    parse.add_argument("--d_model", type=int, default=512)
    parse.add_argument("--d_ff", type=int, default=2048)
    parse.add_argument("--num_layers", type=int, default=6)
    parse.add_argument("--num_heads", type=int, default=8)
    parse.add_argument("--max_seq_len", type=int, default=256)

    # hyper 
    parse.add_argument("--lr", type=float, default=6e-4)
    parse.add_argument("--warmup_iter", type=int, default=1000)
    parse.add_argument("--max_norm", type=float, default=1.0)
    parse.add_argument("--min_lr", type=float, default=6e-5)
    parse.add_argument("--theta", type=int, default=10000)
    parse.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parse.add_argument("--seed", type=int, default=42)
    parse.add_argument("--exp_name", type=str, default=None)

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

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    train(config)
